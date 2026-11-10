#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_contract_features.py
============================
Compiler-free feature extraction for Solidity smart contracts.

Dataset : mwritescode/slither-audited-smart-contracts  (config: small-multilabel)
Input   : /content/smart_contract_data/slither_input.parquet
Outputs : /content/smart_contract_data/extracted/contract_features.parquet
          /content/smart_contract_data/extracted/feature_extraction_errors.csv
          /content/smart_contract_data/extracted/feature_dictionary.csv
          /content/smart_contract_data/extracted/feature_columns.json

NO Slither, NO solc, NO solc-select, NO Hardhat, NO Solidity compiler.
Everything is done with a lexical scanner + bracket matching + regular
expressions on the raw source text.

Design
------
1. LEXER      Comments and string literals are blanked with spaces (newlines are
              kept), so character offsets and line numbers never change and
              keywords inside comments/strings are never counted.
2. STRUCTURE  Brace/parenthesis matching tables are built in one linear pass.
              Contracts, libraries, interfaces, functions, constructors,
              fallback/receive functions and modifiers are located with them
              (multi-line headers and nested braces are handled correctly).
3. FUNCTION   ~85 features per function (see FN_META below).
4. CONTRACT   Function features are aggregated per source file with
              sum / max / mean, plus contract-wide features (see CT_META).

Reliability labels (see feature_dictionary.csv)
-----------------------------------------------
  syntactic : a count/flag of something literally present in the cleaned source
              (e.g. number of `require(`).  Reliable, up to lexer limitations.
  heuristic : an estimate from approximate analysis (e.g. "state written after
              an external call", unchecked-call detection, access-guard
              detection, inherited state variables).  These are ML INPUT
              FEATURES, NOT vulnerability verdicts.  Names end in `_est`
              where the estimate is heuristic.

Leakage safeguards
------------------
* `labels`, `source_hash`, `split`, `duplicate_count`, `original_splits` and the
  `target_*` columns are metadata/targets and are listed separately in
  feature_columns.json. They are never used to compute any feature.
* Your existing `split` column is copied through untouched.

Usage
-----
    pip install pandas numpy pyarrow
    python extract_contract_features.py
    python extract_contract_features.py --workers 1 --limit 50 --batch-size 25
    python extract_contract_features.py --workers 2 --batch-size 250
    # Re-run the same command to resume from completed checkpoints
    # Add --restart only when you intentionally want to discard checkpoints
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths / labels
# ----------------------------------------------------------------------------
DEFAULT_INPUT = "/content/smart_contract_data/slither_input.parquet"
DEFAULT_OUTDIR = "/content/smart_contract_data/extracted"

LABEL_NAMES = {
    0: "access-control", 1: "arithmetic", 2: "other", 3: "reentrancy",
    4: "safe", 5: "unchecked-calls", 6: "locked-ether",
    7: "bad-randomness", 8: "double-spending",
}
TARGET_IDS = [0, 1, 3, 5, 6, 7, 8]          # 'other' (2) and 'safe' (4) are NOT targets


def target_col(label_id: int) -> str:
    return "target_" + LABEL_NAMES[label_id].replace("-", "_")


TARGET_COLS = [target_col(i) for i in TARGET_IDS]

# ----------------------------------------------------------------------------
# Feature registries (name -> (reliability, description))
# ----------------------------------------------------------------------------
FN_META: dict = {}
CT_META: dict = {}
S, H = "syntactic", "heuristic"


def _fn(name, rel, desc):
    FN_META[name] = (rel, desc)


def _ct(name, rel, desc):
    CT_META[name] = (rel, desc)


# ---- function structure
_fn("n_lines", S, "Lines spanned by the function (header + body)")
_fn("n_statements", S, "Semicolon-terminated statements (for-loop header semicolons excluded)")
_fn("n_if", S, "`if` keywords")
_fn("n_else", S, "`else` keywords")
_fn("n_loops", S, "`for` + `while` keywords")
_fn("n_for", S, "`for` keywords")
_fn("n_while", S, "`while` keywords")
_fn("loop_over_length", S, "1 if a loop condition contains `.length` (loop over array length)")
_fn("n_nested_ifs", H, "`if` statements located inside the braces of another if/else block (brace-less ifs not seen)")
_fn("max_if_depth", H, "Maximum nesting depth of if/else blocks (brace-less ifs not seen)")
_fn("max_block_depth", S, "Maximum brace-block depth inside the function body")
# ---- function kind / properties
_fn("is_function", S, "Regular function (not constructor/fallback/receive/modifier)")
_fn("is_constructor", S, "`constructor` keyword, or old-style constructor (function named like its contract)")
_fn("is_fallback", S, "Fallback function (`fallback()` or unnamed `function()`)")
_fn("is_receive", S, "`receive()` function")
_fn("is_modifier", S, "Modifier definition")
_fn("in_library", S, "Function defined inside a `library`")
_fn("vis_public", S, "Explicit `public`")
_fn("vis_private", S, "Explicit `private`")
_fn("vis_internal", S, "Explicit `internal`")
_fn("vis_external", S, "Explicit `external`")
_fn("vis_missing", S, "No explicit visibility (legacy Solidity < 0.5 defaults to public)")
_fn("eff_public", H, "Effectively public: explicit public, or visibility omitted on a callable function")
_fn("is_payable", S, "`payable`")
_fn("is_view", S, "`view` or `constant`")
_fn("is_pure", S, "`pure`")
_fn("n_modifiers_used", H, "Modifier/base-constructor invocations in the header (identifiers that are not keywords)")
_fn("has_access_modifier", H, "A used modifier name looks like access control (only*/owner/admin/auth/role...)")
_fn("has_reentrancy_guard_modifier", H, "A used modifier name looks like a reentrancy guard (nonReentrant/lock/mutex...)")
# ---- external interactions
_fn("n_call", S, "`.call` uses")
_fn("n_delegatecall", S, "`.delegatecall` uses")
_fn("n_staticcall", S, "`.staticcall` uses")
_fn("n_callcode", S, "`.callcode` uses")
_fn("n_send", S, "`.send(` uses")
_fn("n_transfer", S, "`.transfer(` uses (ETH transfers AND token transfers)")
_fn("n_eth_transfer_calls", H, "`.transfer(` with exactly one argument (ETH transfer, not token transfer)")
_fn("n_value_calls", S, "Calls carrying ETH value: `.value(` or `{value: ...}`")
_fn("n_low_level_calls", S, "call + delegatecall + staticcall + callcode + send")
_fn("n_ext_interactions", S, "low-level calls + `.transfer(`")
_fn("n_ether_out_ops_est", H, "Ways ETH can leave: send, 1-arg transfer, value calls, selfdestruct")
_fn("n_unchecked_call_like_est", H, "call/delegatecall/staticcall/callcode statements whose result is not consumed")
_fn("n_unchecked_send_est", H, "`.send` statements whose result is not consumed")
_fn("n_unchecked_low_level_est", H, "Unchecked call-like + unchecked send")
_fn("ext_call_in_loop_est", H, "External interaction textually inside a for/while body")
# ---- state interaction (state variables found by declaration parsing; shadowing ignored)
_fn("n_state_reads_est", H, "Approx. occurrences of state-variable names that are not plain-assignment targets")
_fn("n_state_writes_est", H, "Approx. writes to state variables (=, op=, ++/--, delete, push/pop)")
_fn("n_state_writes_after_ext_call_est", H, "State writes textually after the first external interaction")
_fn("state_write_after_ext_call_est", H, "1 if any state write follows the first external interaction")
_fn("state_write_before_ext_call_est", H, "1 if a state write precedes the first external interaction")
_fn("state_write_after_value_call_est", H, "1 if a state write follows the first ETH-value call")
# ---- security-related tokens
_fn("n_require", S, "`require(` uses")
_fn("n_assert", S, "`assert(` uses")
_fn("n_revert", S, "`revert` + `throw` uses")
_fn("n_checks", S, "require + assert + revert")
_fn("n_msg_sender", S, "`msg.sender` uses")
_fn("n_msg_sender_cmp", S, "`msg.sender ==/!=` comparisons")
_fn("n_auth_check_msg_sender_est", H, "require/assert/if whose condition mentions msg.sender")
_fn("n_tx_origin", S, "`tx.origin` uses")
_fn("n_tx_origin_cmp", S, "`tx.origin ==/!=` comparisons")
_fn("n_msg_value", S, "`msg.value` uses")
_fn("n_block_timestamp", S, "`block.timestamp` + `now`")
_fn("n_block_number", S, "`block.number`")
_fn("n_blockhash", S, "`blockhash` / `block.blockhash`")
_fn("n_block_other", S, "block.difficulty/prevrandao/coinbase/gaslimit/basefee")
_fn("n_selfdestruct", S, "`selfdestruct` + `suicide`")
_fn("n_keccak", S, "keccak256 + sha3 + sha256")
_fn("n_assembly", S, "`assembly` keyword")
_fn("n_safemath_ident", S, "Identifier `SafeMath`")
_fn("n_safemath_like_calls", H, "`.add( .sub( .mul( .div( .mod(` member calls")
_fn("n_ecrecover", S, "`ecrecover` uses")
_fn("n_nonce_ident", H, "Identifiers containing 'nonce'")
_fn("n_owner_admin_ident", H, "Identifiers containing 'owner' or 'admin'")
_fn("n_balance_access", S, "`.balance` member accesses")
_fn("n_unchecked_blocks", S, "`unchecked` keyword (Solidity 0.8 unchecked arithmetic blocks)")
_fn("n_emit", S, "`emit` statements")
_fn("block_attr_in_hash_est", H, "Block attribute and a hash function co-occur in the function")
_fn("block_attr_with_modulo_est", H, "Block attribute and `%` co-occur in the function")
_fn("public_state_write_no_guard_est", H, "Exposed function writes state with no access-check pattern found")
_fn("selfdestruct_no_guard_est", H, "Exposed function uses selfdestruct with no access-check pattern found")
_fn("delegatecall_no_guard_est", H, "Exposed function uses delegatecall with no access-check pattern found")
# ---- arithmetic
_fn("n_add", H, "Binary `+` (unary plus excluded by context check)")
_fn("n_sub", H, "Binary `-` (unary minus excluded by context check)")
_fn("n_mul", S, "`*`")
_fn("n_div", S, "`/`")
_fn("n_mod", S, "`%`")
_fn("n_exp", S, "`**`")
_fn("n_inc_dec", S, "`++` and `--`")
_fn("n_arith_compound", S, "`+= -= *= /= %=`")
_fn("n_compound_assign", S, "All compound assignments (incl. bitwise and shifts)")
_fn("n_arith_total", S, "add+sub+mul+div+mod+exp+inc_dec+arith_compound")

FN_NAMES = list(FN_META)
AGGS = ("sum", "max", "mean")

# ---- contract-level
_ct("c_n_contracts", S, "`contract` declarations (incl. abstract)")
_ct("c_n_abstract_contracts", S, "`abstract contract` declarations")
_ct("c_n_libraries", S, "`library` declarations")
_ct("c_n_interfaces", S, "`interface` declarations")
_ct("c_n_functions", S, "Regular functions with a body")
_ct("c_n_constructors", S, "Constructors (keyword or old-style)")
_ct("c_n_modifiers_defined", S, "Modifier definitions")
_ct("c_n_fallback", S, "Fallback functions")
_ct("c_n_receive", S, "Receive functions")
_ct("c_n_bodyless_function_decls", S, "Function declarations without a body (interfaces/abstract)")
_ct("c_n_state_vars_est", H, "State-variable declarations found by top-level statement parsing")
_ct("c_n_events", S, "`event` keywords")
_ct("c_n_structs", S, "`struct` keywords")
_ct("c_n_enums", S, "`enum` keywords")
_ct("c_n_mappings", S, "`mapping` keywords")
_ct("c_n_using_for", S, "`using` keywords")
_ct("c_n_imports", S, "`import` keywords")
_ct("c_n_inheritance", S, "Total inheritance relationships (sum of base-contract lists)")
_ct("c_max_inheritance", S, "Largest number of bases of a single contract")
_ct("c_n_pragma_solidity", S, "`pragma solidity` statements")
_ct("solc_missing", S, "No `pragma solidity` found")
_ct("solc_major", S, "Effective compiler major version from pragma (-1 if missing)")
_ct("solc_minor", S, "Effective compiler minor version from pragma (-1 if missing)")
_ct("solc_patch", S, "Effective compiler patch version from pragma (-1 if missing)")
_ct("solc_version_num", S, "major*10000 + minor*100 + patch (-1 if missing)")
_ct("solc_below_0_8", S, "1 if the pragma's lowest allowed version is < 0.8.0 (no built-in overflow checks)")
_ct("solc_pinned", S, "1 if every pragma pins an exact version (no ^ > < ~)")
_ct("c_uses_safemath", S, "Identifier `SafeMath` present")
_ct("c_uses_assembly", S, "`assembly` present")
_ct("c_n_assembly", S, "`assembly` keyword count")
_ct("c_src_chars", S, "Source length in characters")
_ct("c_src_lines", S, "Source line count")
_ct("c_code_lines", S, "Non-blank lines after removing comments and strings")
_ct("c_comment_ratio", S, "Comment characters / total characters")
_ct("c_no_functions_detected", S, "1 if no function with a body was found")
_ct("c_total_ext_interactions", S, "Whole-file count of .call/.delegatecall/.staticcall/.callcode/.send/.transfer(")
_ct("c_total_low_level_calls", S, "Whole-file count of low-level calls")
_ct("c_total_arith_ops", S, "Whole-file arithmetic operator count (incl. outside functions)")
_ct("c_total_checks", S, "Whole-file require + assert + revert + throw")
_ct("c_total_payable_functions", H, "Functions/fallback/receive marked payable")
_ct("c_total_external_functions", S, "Functions with explicit `external`")
_ct("c_total_public_functions", H, "Effectively public functions")
_ct("c_has_payable_entry", H, "Any payable function, or payable fallback/receive")
_ct("c_has_ether_out", H, "Any way for ETH to leave (send, 1-arg transfer, value call, selfdestruct)")
_ct("c_payable_no_ether_out_est", H, "Can receive ETH but no ETH-out operation found (locked-ether pattern)")
_ct("c_arith_unprotected_est", H, "Arithmetic present, compiler < 0.8, and no SafeMath-like usage")
_ct("file_n_tx_origin", S, "Whole-file `tx.origin` count")
_ct("file_n_delegatecall", S, "Whole-file `.delegatecall` count")
_ct("file_n_selfdestruct", S, "Whole-file selfdestruct/suicide count")
_ct("file_n_block_timestamp", S, "Whole-file `block.timestamp` + `now` count")
_ct("file_n_blockhash", S, "Whole-file blockhash count")
_ct("file_n_ecrecover", S, "Whole-file ecrecover count")
CT_NAMES = list(CT_META)

# ----------------------------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------------------------
IDENT = r"[A-Za-z_$][\w$]*"
INDEX = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"          # one level of nested []
_ACC = rf"(?:\s*{INDEX}|\s*\.\s*{IDENT})*"

LEX_RE = re.compile(
    r"//[^\n]*"                      # line comment
    r"|/\*.*?\*/"                    # block comment
    r"|/\*.*\Z"                      # unterminated block comment (to EOF)
    r'|"(?:\\.|[^"\\\n])*"'          # double-quoted string
    r"|'(?:\\.|[^'\\\n])*'",         # single-quoted string
    re.S,
)
CALL_OPT_RE = re.compile(r"\{(\s*(?:value|gas|salt)\s*:[^{}]*)\}")      # x.call{value: v}(...)
BRACKET_RE = re.compile(r"[(){}]")
WORD_RE = re.compile(IDENT)
# Token-aware counts for Solidity constructs (comments/strings already blanked).
def count_call(src: str, name: str) -> int:
    return len(re.findall(r"\b" + re.escape(name) + r"\s*\(", src))

REQUIRE_CALL_RE = re.compile(r"\brequire\s*\(")
ASSERT_CALL_RE = re.compile(r"\bassert\s*\(")
REVERT_RE = re.compile(r"\brevert\b(?:\s*\()?" )
EMIT_RE = re.compile(r"\bemit\s+[A-Za-z_$][\w$]*\s*\(")
ASSEMBLY_RE = re.compile(r"\bassembly\b")
SELFDESTRUCT_RE = re.compile(r"\b(?:selfdestruct|suicide)\s*\(")
BLOCKHASH_RE = re.compile(r"\bblockhash\s*\(")
CONTRACT_RE = re.compile(
    r"\b(?P<abs>abstract\s+)?(?P<kind>contract|library|interface)\s+(?P<name>" + IDENT + r")"
    r"\s*(?P<is>\bis\b[^{;]*)?\{"
)
FUNC_RE = re.compile(r"\b(function|constructor|modifier|fallback|receive)\b")
HEADER_END_RE = re.compile(r"[{;]")
PRAGMA_RE = re.compile(r"\bpragma\s+solidity\b([^;]*);")
VER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
NON_VAR_START = re.compile(
    r"^(function|modifier|constructor|fallback|receive|event|error|struct|enum|using|pragma|"
    r"import|contract|library|interface|abstract|emit|assembly|type)\b"
)

EXT_RE = re.compile(r"\.\s*(call|delegatecall|staticcall|callcode|send|transfer)\b")
LOW_CALL_RE = re.compile(r"\.\s*(call|delegatecall|staticcall|callcode|send)\b")
VALUE_RE = re.compile(r"\.\s*value\s*\(|\(\s*value\s*:")
LOOP_RE = re.compile(r"\b(for|while)\s*\(")
IFWORD_RE = re.compile(r"\b(if|else)\b")
EVENT_RE = re.compile(r"\bif\b|[{}]")
LENGTH_RE = re.compile(r"\.\s*length\b")
MSG_SENDER_RE = re.compile(r"\bmsg\s*\.\s*sender\b")
MSG_VALUE_RE = re.compile(r"\bmsg\s*\.\s*value\b")
MSG_SENDER_CMP_RE = re.compile(r"\bmsg\s*\.\s*sender\s*[!=]=|[!=]=\s*msg\s*\.\s*sender\b")
AUTH_MSG_RE = re.compile(r"\b(?:require|assert|if)\s*\([^;{}]*\bmsg\s*\.\s*sender\b")
TX_ORIGIN_RE = re.compile(r"\btx\s*\.\s*origin\b")
TX_ORIGIN_CMP_RE = re.compile(r"\btx\s*\.\s*origin\s*[!=]=|[!=]=\s*tx\s*\.\s*origin\b")
BLOCK_RE = re.compile(r"\bblock\s*\.\s*(timestamp|number|difficulty|prevrandao|coinbase|gaslimit|basefee)\b")
BALANCE_RE = re.compile(r"\.\s*balance\b")
SAFEMATH_CALL_RE = re.compile(r"\.\s*(add|sub|mul|div|mod)\s*\(")
OP_RE = re.compile(r"\+\+|--|\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=|\*\*|[+\-*/%]")
ARITH_COMPOUND = {"+=", "-=", "*=", "/=", "%="}
OTHER_COMPOUND = {"<<=", ">>=", "&=", "|=", "^="}

WRITE_RE = re.compile(rf"(?<![\w$.])(?P<base>{IDENT}){_ACC}\s*(?P<op>[+\-*/%&|^]|<<|>>)?=(?![=>])")
PREINC_RE = re.compile(rf"(?:\+\+|--)\s*(?P<base>{IDENT})")
POSTINC_RE = re.compile(rf"(?<![\w$.])(?P<base>{IDENT}){_ACC}\s*(?:\+\+|--)")
DEL_RE = re.compile(rf"\bdelete\s+(?P<base>{IDENT})")
PUSH_RE = re.compile(rf"(?<![\w$.])(?P<base>{IDENT}){_ACC}\s*\.\s*(?:push|pop)\s*\(")

STMT_SPLIT_RE = re.compile(r"[;{}]")
CTRL_PREFIX_RE = re.compile(r"(?:if|while|for)\s*\(")
CONSUMED_PREFIX_RE = re.compile(r"\b(?:require|assert|return)\b|(?<![=!<>])=(?![=>])|!|\?")
CONSUMED_ANY_RE = re.compile(r"&&|\|\|")

ACCESS_MOD_RE = re.compile(
    r"(?i)only|owner|admin|auth|role|restrict|governor|governance|operator|minter|manager|guardian"
)
GUARD_MOD_RE = re.compile(r"(?i)nonreentrant|noreentr|reentranc|mutex|\block\b|locked|guard")
HEADER_RESERVED = {
    "public", "private", "internal", "external", "payable", "view", "pure", "constant",
    "virtual", "override", "returns", "immutable", "function", "fallback", "receive",
    "constructor", "modifier",
}
NESTED_PARENS_RE = re.compile(r"\([^()]*\)")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def next_nonspace(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return i


def local_match(s: str, open_idx: int) -> int:
    """Index of the ')' matching the '(' at open_idx inside a small string, else -1."""
    depth = 0
    for k in range(open_idx, len(s)):
        c = s[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def top_level_commas(s: str) -> int:
    depth = n = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def split_top_commas(s: str) -> list:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------------------
# 1. Lexer: blank comments and strings, keep offsets and newlines
# ----------------------------------------------------------------------------
def clean_source(src: str):
    """Return (clean_source, n_comment_chars). Same length as `src`."""
    acc = [0]

    def blank(m):
        s = m.group(0)
        if s.startswith("/"):
            acc[0] += len(s)
        if "\n" not in s:
            return " " * len(s)
        return "\n".join(" " * len(p) for p in s.split("\n"))

    clean = LEX_RE.sub(blank, src)
    # `{value: x}` call options are not code blocks: turn braces into parens (same length)
    clean = CALL_OPT_RE.sub(lambda m: "(" + m.group(1) + ")", clean)
    return clean, acc[0]


# ----------------------------------------------------------------------------
# 2. Bracket matching
# ----------------------------------------------------------------------------
def bracket_maps(clean: str):
    brace_close, paren_close = {}, {}
    bstack, pstack = [], []
    stray = 0
    for m in BRACKET_RE.finditer(clean):
        ch, i = m.group(), m.start()
        if ch == "{":
            bstack.append(i)
        elif ch == "}":
            if bstack:
                brace_close[bstack.pop()] = i
            else:
                stray += 1
        elif ch == "(":
            pstack.append(i)
        else:
            if pstack:
                paren_close[pstack.pop()] = i
            else:
                stray += 1
    return brace_close, paren_close, len(bstack) + len(pstack) + stray


# ----------------------------------------------------------------------------
# Contract declarations and state variables
# ----------------------------------------------------------------------------
def find_contracts(clean: str, brace_close: dict, warnings: list):
    contracts = []
    for m in CONTRACT_RE.finditer(clean):
        open_idx = m.end() - 1
        close = brace_close.get(open_idx)
        if close is None:
            warnings.append(("unmatched_contract_brace", f"contract {m.group('name')}"))
            close = len(clean)
        is_clause = m.group("is")
        bases = []
        if is_clause:
            for part in split_top_commas(is_clause.strip()[2:]):
                ids = WORD_RE.findall(part.split("(")[0])
                if ids:
                    bases.append(ids[-1])                 # `A.B` -> B ; `Ownable(msg.sender)` -> Ownable
        contracts.append({
            "kind": m.group("kind"), "name": m.group("name"), "abstract": bool(m.group("abs")),
            "start": m.start(), "open": open_idx, "close": close, "bases": bases,
            "state_vars": set(), "n_state_vars": 0,
        })
    return contracts


def parse_state_vars(clean: str, rec: dict, brace_close: dict) -> None:
    """Top-level statements of a contract body that are not functions/events/etc."""
    pieces, i, end = [], rec["open"] + 1, min(rec["close"], len(clean))
    while True:
        nb = clean.find("{", i, end)
        if nb == -1:
            pieces.append(clean[i:end])
            break
        seg = clean[i:nb]
        pieces.append(seg)
        cb = brace_close.get(nb)
        if cb is None:
            pieces.append(";")
            break
        prev = seg.rstrip()[-1:]
        if prev not in ("(", ",", ":"):      # struct literals inside initialisers: not a block
            pieces.append(";")
        i = cb + 1
    names = set()
    n = 0
    for stmt in "".join(pieces).split(";"):
        s = stmt.strip()
        if not s or NON_VAR_START.match(s):
            continue
        decl = re.split(r"=(?!>)", s, maxsplit=1)[0]
        ids = WORD_RE.findall(decl)
        if len(ids) >= 2:
            names.add(ids[-1])
            n += 1
    rec["state_vars"] = names
    rec["n_state_vars"] = n


def inherited_state_vars(rec: dict, by_name: dict, seen=None) -> set:
    seen = seen or set()
    if rec["name"] in seen:
        return set()
    seen.add(rec["name"])
    out = set(rec["state_vars"])
    for b in rec["bases"]:
        parent = by_name.get(b)
        if parent is not None:
            out |= inherited_state_vars(parent, by_name, seen)
    return out


# ----------------------------------------------------------------------------
# Pragma
# ----------------------------------------------------------------------------
def parse_pragma(clean: str) -> dict:
    """Conservative pragma parser. Only exact pins produce a concrete version.

    Ranges such as ^0.7.0, >=0.7.0 <0.9.0 and || alternatives are ambiguous
    without resolving the actual compiler used, so compiler-dependent features
    are disabled for those files (solc_below_0_8 = -1).
    """
    pragmas = list(PRAGMA_RE.finditer(clean))
    out = {"c_n_pragma_solidity": len(pragmas), "solc_missing": int(not pragmas),
           "solc_major": -1, "solc_minor": -1, "solc_patch": -1,
           "solc_version_num": -1, "solc_below_0_8": -1, "solc_pinned": 0}
    if not pragmas:
        return out

    exact_versions = []
    all_exact = True
    for m in pragmas:
        spec = m.group(1).strip()
        # Exact pin: optional '=' followed by one full semantic version only.
        exact = re.fullmatch(r"=?\s*(\d+)\.(\d+)\.(\d+)\s*", spec)
        if not exact:
            all_exact = False
            break
        exact_versions.append(tuple(map(int, exact.groups())))

    if all_exact and exact_versions and len(set(exact_versions)) == 1:
        major, minor, patch = exact_versions[0]
        out.update(solc_missing=0, solc_major=major, solc_minor=minor,
                   solc_patch=patch, solc_version_num=major*10000+minor*100+patch,
                   solc_below_0_8=int((major, minor, patch) < (0, 8, 0)),
                   solc_pinned=1)
    return out


# ----------------------------------------------------------------------------
# Unchecked low-level call heuristic
# ----------------------------------------------------------------------------
def count_unchecked_calls(body: str):
    """Statements where the result of a low-level call/send is not consumed."""
    call_like = sends = 0
    for stmt in STMT_SPLIT_RE.split(body):
        if not LOW_CALL_RE.search(stmt):
            continue
        s = stmt.strip()
        for _ in range(8):                                   # peel else / if(...) / while(...) / for(...)
            if re.match(r"else\b", s):
                s = s[4:].lstrip()
                continue
            m = CTRL_PREFIX_RE.match(s)
            if m:
                close = local_match(s, m.end() - 1)
                if close == -1:
                    s = ""
                    break
                s = s[close + 1:].lstrip()
                continue
            break
        if not s or CONSUMED_ANY_RE.search(s):
            continue
        for m in LOW_CALL_RE.finditer(s):
            kind = m.group(1)
            if kind == "send":
                j = next_nonspace(s, m.end())
                if j >= len(s) or s[j] != "(":
                    continue
            if CONSUMED_PREFIX_RE.search(s[:m.start()]):
                continue
            if kind == "send":
                sends += 1
            else:
                call_like += 1
    return call_like, sends


# ----------------------------------------------------------------------------
# 3. Function-level features
# ----------------------------------------------------------------------------
def parse_header(header: str, kind: str):
    """Return (modifier_names, attribute_words).

    Attributes (public/payable/view/...) are read ONLY from the part of the header after the
    parameter list and outside returns(...), so `address payable to` in a parameter list does
    not make a function payable."""
    p = header.find("(")
    if p == -1:
        return [], set()
    close = local_match(header, p)
    if close == -1:
        return [], set()
    rest = header[close + 1:]
    k = rest.find("returns")
    if k != -1:                                              # drop the returns(...) clause
        q = rest.find("(", k)
        if q != -1:
            c2 = local_match(rest, q)
            rest = rest[:k] + (rest[c2 + 1:] if c2 != -1 else "")
    for _ in range(4):                                       # strip argument lists, innermost first
        new = NESTED_PARENS_RE.sub(" ", rest)
        if new == rest:
            break
        rest = new
    words = WORD_RE.findall(rest)
    return [w for w in words if w not in HEADER_RESERVED], set(words)


def analyze_function(clean, brace_close, paren_close, fs, hdr_end, fc,
                     name, kind, contract, state_vars, solc_below_08):
    f = dict.fromkeys(FN_NAMES, 0)
    header = clean[fs:hdr_end]
    off = hdr_end + 1                      # absolute offset of the body text
    body = clean[off:fc]
    words = Counter(WORD_RE.findall(body))
    mods, hw = parse_header(header, kind)
    wget = words.get

    # ---- kind / properties
    old_ctor = kind == "function" and contract is not None and contract["kind"] != "interface" \
        and name == contract["name"]
    f["is_constructor"] = int(kind == "constructor" or old_ctor)
    f["is_fallback"] = int(kind == "fallback")
    f["is_receive"] = int(kind == "receive")
    f["is_modifier"] = int(kind == "modifier")
    f["is_function"] = int(kind == "function" and not old_ctor)
    f["in_library"] = int(contract is not None and contract["kind"] == "library")
    f["vis_public"] = int("public" in hw)
    f["vis_private"] = int("private" in hw)
    f["vis_internal"] = int("internal" in hw)
    f["vis_external"] = int("external" in hw)
    f["vis_missing"] = int(not (hw & {"public", "private", "internal", "external"}))
    f["eff_public"] = int(f["vis_public"] or (f["vis_missing"] and kind != "modifier"))
    f["is_payable"] = int("payable" in hw)
    f["is_view"] = int(bool(hw & {"view", "constant"}))
    f["is_pure"] = int("pure" in hw)
    f["n_modifiers_used"] = len(mods)
    f["has_access_modifier"] = int(any(ACCESS_MOD_RE.search(w) for w in mods))
    f["has_reentrancy_guard_modifier"] = int(any(GUARD_MOD_RE.search(w) for w in mods))

    # ---- structure
    f["n_lines"] = clean.count("\n", fs, fc + 1) + 1
    f["n_if"] = wget("if", 0)
    f["n_else"] = wget("else", 0)
    f["n_for"] = wget("for", 0)
    f["n_while"] = wget("while", 0)
    f["n_loops"] = f["n_for"] + f["n_while"]

    semis = body.count(";")
    loop_ranges = []
    if f["n_loops"]:
        for m in LOOP_RE.finditer(body):
            o = off + m.end() - 1
            c = paren_close.get(o)
            if c is None:
                continue
            if m.group(1) == "for":
                semis -= clean.count(";", o, c)
            if LENGTH_RE.search(clean, o, c):
                f["loop_over_length"] = 1
            j = next_nonspace(clean, c + 1)
            if j < len(clean) and clean[j] == "{":
                loop_ranges.append((j, brace_close.get(j, fc)))
            else:
                e = clean.find(";", j, fc)
                loop_ranges.append((j, e if e != -1 else fc))
    f["n_statements"] = max(semis, 0)

    if_open = set()
    if f["n_if"] or f["n_else"]:
        for m in IFWORD_RE.finditer(body):
            if m.group(1) == "if":
                k = next_nonspace(clean, off + m.end())
                if k < len(clean) and clean[k] == "(":
                    c = paren_close.get(k)
                    if c is not None:
                        j = next_nonspace(clean, c + 1)
                        if j < len(clean) and clean[j] == "{":
                            if_open.add(j)
            else:
                j = next_nonspace(clean, off + m.end())
                if j < len(clean) and clean[j] == "{":
                    if_open.add(j)
    stack, true_cnt, maxd, nested, maxblk = [], 0, 0, 0, 0
    for m in EVENT_RE.finditer(body):
        tok = m.group()
        if tok == "{":
            t = (off + m.start()) in if_open
            stack.append(t)
            true_cnt += t
            maxblk = max(maxblk, len(stack))
        elif tok == "}":
            if stack:
                true_cnt -= stack.pop()
        else:
            maxd = max(maxd, true_cnt + 1)
            if true_cnt > 0:
                nested += 1
    f["n_nested_ifs"], f["max_if_depth"], f["max_block_depth"] = nested, maxd, maxblk

    # ---- external interactions
    ext_pos, eth_transfer = [], 0
    if any(wget(w) for w in ("call", "delegatecall", "staticcall", "callcode", "send", "transfer")):
        for m in EXT_RE.finditer(body):
            k = m.group(1)
            if k in ("send", "transfer"):
                j = next_nonspace(clean, off + m.end())
                if j >= len(clean) or clean[j] != "(":
                    continue
                if k == "transfer":
                    c = paren_close.get(j)
                    if c is not None:
                        inner = clean[j + 1:c]
                        if inner.strip() and top_level_commas(inner) == 0:
                            eth_transfer += 1
            f["n_" + k] += 1
            ext_pos.append(m.start())
    f["n_eth_transfer_calls"] = eth_transfer
    f["n_low_level_calls"] = (f["n_call"] + f["n_delegatecall"] + f["n_staticcall"]
                              + f["n_callcode"] + f["n_send"])
    f["n_ext_interactions"] = f["n_low_level_calls"] + f["n_transfer"]

    value_pos = []
    if wget("value"):
        value_pos = [m.start() for m in VALUE_RE.finditer(body)]
    f["n_value_calls"] = len(value_pos)

    f["n_selfdestruct"] = len(SELFDESTRUCT_RE.findall(body))
    f["n_ether_out_ops_est"] = (f["n_send"] + eth_transfer + f["n_value_calls"] + f["n_selfdestruct"])

    if f["n_call"] or f["n_delegatecall"] or f["n_staticcall"] or f["n_callcode"] or f["n_send"]:
        cl, sd = count_unchecked_calls(body)
        f["n_unchecked_call_like_est"], f["n_unchecked_send_est"] = cl, sd
        f["n_unchecked_low_level_est"] = cl + sd
    if ext_pos and loop_ranges:
        for p in ext_pos:
            a = off + p
            if any(lo <= a <= hi for lo, hi in loop_ranges):
                f["ext_call_in_loop_est"] = 1
                break

    # ---- state interaction (heuristic: names of state variables, shadowing ignored)
    hit = words.keys() & state_vars if state_vars else set()
    if hit:
        occ = sum(words[v] for v in hit)
        writes, simple_writes = [], 0
        for m in WRITE_RE.finditer(body):
            if m.group("base") in hit:
                writes.append(m.start())
                if not m.group("op"):
                    simple_writes += 1
        if "++" in body or "--" in body:
            for rx in (PREINC_RE, POSTINC_RE):
                for m in rx.finditer(body):
                    if m.group("base") in hit:
                        writes.append(m.start())
        if wget("delete"):
            for m in DEL_RE.finditer(body):
                if m.group("base") in hit:
                    writes.append(m.start())
                    simple_writes += 1
        if wget("push") or wget("pop"):
            for m in PUSH_RE.finditer(body):
                if m.group("base") in hit:
                    writes.append(m.start())
        f["n_state_writes_est"] = len(writes)
        f["n_state_reads_est"] = max(occ - simple_writes, 0)
        if ext_pos:
            first = min(ext_pos)
            after = sum(1 for w in writes if w > first)
            f["n_state_writes_after_ext_call_est"] = after
            f["state_write_after_ext_call_est"] = int(after > 0)
            f["state_write_before_ext_call_est"] = int(any(w < first for w in writes))
        if value_pos:
            fv = min(value_pos)
            f["state_write_after_value_call_est"] = int(any(w > fv for w in writes))

    # ---- security tokens
    f["n_require"] = len(REQUIRE_CALL_RE.findall(body))
    f["n_assert"] = len(ASSERT_CALL_RE.findall(body))
    f["n_revert"] = len(REVERT_RE.findall(body)) + len(re.findall(r"\bthrow\b", body))
    f["n_checks"] = f["n_require"] + f["n_assert"] + f["n_revert"]
    if wget("msg"):
        f["n_msg_sender"] = len(MSG_SENDER_RE.findall(body))
        f["n_msg_value"] = len(MSG_VALUE_RE.findall(body))
        f["n_msg_sender_cmp"] = len(MSG_SENDER_CMP_RE.findall(body))
        if f["n_msg_sender"] and (f["n_require"] or f["n_assert"] or f["n_if"]):
            f["n_auth_check_msg_sender_est"] = len(AUTH_MSG_RE.findall(body))
    if wget("tx"):
        f["n_tx_origin"] = len(TX_ORIGIN_RE.findall(body))
        f["n_tx_origin_cmp"] = len(TX_ORIGIN_CMP_RE.findall(body))
    ts = len(re.findall(r"\bnow\b", body))
    if wget("block"):
        for m in BLOCK_RE.finditer(body):
            a = m.group(1)
            if a == "timestamp":
                ts += 1
            elif a == "number":
                f["n_block_number"] += 1
            else:
                f["n_block_other"] += 1
    f["n_block_timestamp"] = ts
    f["n_blockhash"] = len(BLOCKHASH_RE.findall(body))
    f["n_keccak"] = wget("keccak256", 0) + wget("sha3", 0) + wget("sha256", 0)
    f["n_assembly"] = len(ASSEMBLY_RE.findall(body))
    f["n_safemath_ident"] = wget("SafeMath", 0)
    if any(wget(w) for w in ("add", "sub", "mul", "div", "mod")):
        f["n_safemath_like_calls"] = len(SAFEMATH_CALL_RE.findall(body))
    f["n_ecrecover"] = wget("ecrecover", 0)
    f["n_unchecked_blocks"] = wget("unchecked", 0)
    f["n_emit"] = len(EMIT_RE.findall(body))
    if wget("balance"):
        f["n_balance_access"] = len(BALANCE_RE.findall(body))
    f["n_nonce_ident"] = sum(c for w, c in words.items() if "nonce" in w.lower())
    f["n_owner_admin_ident"] = sum(
        c for w, c in words.items() if "owner" in w.lower() or "admin" in w.lower())

    # ---- arithmetic
    cnt = {"+": 0, "-": 0, "*": 0, "/": 0, "%": 0}
    inc = exp = arith_c = comp = 0
    for m in OP_RE.finditer(body):
        t = m.group()
        if t in ("++", "--"):
            inc += 1
        elif t in ARITH_COMPOUND:
            arith_c += 1
            comp += 1
        elif t in OTHER_COMPOUND:
            comp += 1
        elif t == "**":
            exp += 1
        else:
            if t in ("+", "-"):                               # binary only if an operand precedes it
                j = m.start() - 1
                while j >= 0 and body[j] in " \t\r\n":
                    j -= 1
                if j < 0 or not (body[j].isalnum() or body[j] in "_$)]"):
                    continue
            cnt[t] += 1
    f["n_add"], f["n_sub"], f["n_mul"], f["n_div"], f["n_mod"] = (
        cnt["+"], cnt["-"], cnt["*"], cnt["/"], cnt["%"])
    f["n_exp"], f["n_inc_dec"] = exp, inc
    f["n_arith_compound"], f["n_compound_assign"] = arith_c, comp
    f["n_arith_total"] = sum(cnt.values()) + exp + inc + arith_c

    # ---- co-occurrence heuristics
    block_attr = f["n_block_timestamp"] + f["n_block_number"] + f["n_blockhash"] + f["n_block_other"]
    f["block_attr_in_hash_est"] = int(block_attr > 0 and f["n_keccak"] > 0)
    f["block_attr_with_modulo_est"] = int(block_attr > 0 and f["n_mod"] > 0)
    guarded = (f["has_access_modifier"] or f["n_msg_sender_cmp"] or f["n_tx_origin_cmp"]
               or f["n_auth_check_msg_sender_est"])
    exposed = (f["eff_public"] or f["vis_external"]) and kind in ("function", "fallback", "receive") \
        and not f["is_constructor"]
    f["public_state_write_no_guard_est"] = int(bool(exposed and f["n_state_writes_est"] > 0 and not guarded))
    f["selfdestruct_no_guard_est"] = int(bool(exposed and f["n_selfdestruct"] > 0 and not guarded))
    f["delegatecall_no_guard_est"] = int(bool(exposed and f["n_delegatecall"] > 0 and not guarded))
    return f


# ----------------------------------------------------------------------------
# 4. Whole-source extraction
# ----------------------------------------------------------------------------
def extract_features(source_code):
    """Return (feature_dict, warnings). Raises only on genuinely unexpected bugs."""
    warnings = []
    if source_code is None or (isinstance(source_code, float) and math.isnan(source_code)):
        source_code = ""
    src = source_code if isinstance(source_code, str) else str(source_code)
    if not src.strip():
        warnings.append(("empty_source", "source_code is empty or whitespace"))

    clean, comment_chars = clean_source(src)
    brace_close, paren_close, unbalanced = bracket_maps(clean)
    if unbalanced:
        warnings.append(("unbalanced_brackets", f"{unbalanced} unmatched bracket(s)"))

    ct = dict.fromkeys(CT_NAMES, 0)
    ct.update(parse_pragma(clean))
    if ct["solc_missing"] and src.strip():
        warnings.append(("no_pragma", "no `pragma solidity` statement found"))

    contracts = find_contracts(clean, brace_close, warnings)
    by_name = {}
    for rec in contracts:
        parse_state_vars(clean, rec, brace_close)
        by_name[rec["name"]] = rec
    starts = [c["start"] for c in contracts]

    def contract_at(pos):
        i = bisect.bisect_right(starts, pos) - 1
        if i >= 0 and pos < contracts[i]["close"]:
            return contracts[i]
        return None

    sv_cache = {}

    def state_vars_for(rec):
        if rec is None:
            return set()
        if rec["name"] not in sv_cache:
            sv_cache[rec["name"]] = inherited_state_vars(rec, by_name)
        return sv_cache[rec["name"]]

    # ---- find functions
    fn_rows = []
    n_bodyless = 0
    skip_until = 0
    solc_below = ct["solc_below_0_8"]
    for m in FUNC_RE.finditer(clean):
        start = m.start()
        if start < skip_until:
            continue
        kw = m.group(1)
        j = next_nonspace(clean, m.end())
        nxt = clean[j] if j < len(clean) else ""
        if kw in ("constructor", "fallback", "receive"):
            if nxt != "(":
                continue
            k = start - 1
            while k >= 0 and clean[k] in " \t\r\n":
                k -= 1
            if k >= 0 and clean[k] not in "{};":              # e.g. a call named receive(...)
                continue
        hm = HEADER_END_RE.search(clean, m.end())
        if hm is None:                  # no '{' or ';' anywhere after this point -> none for later keywords either
            break
        hdr_end = hm.start()
        if clean[hdr_end] == ";":
            n_bodyless += 1
            continue
        fc = brace_close.get(hdr_end)
        if fc is None:
            warnings.append(("unmatched_function_brace", f"{kw} at offset {start}"))
            continue
        skip_until = fc + 1
        if kw == "function":
            if nxt == "(":
                kind, name = "fallback", "<fallback>"
            else:
                nm = WORD_RE.match(clean, j)
                kind, name = "function", (nm.group() if nm else "<anonymous>")
        elif kw == "modifier":
            nm = WORD_RE.match(clean, j)
            kind, name = "modifier", (nm.group() if nm else "<anonymous>")
        else:
            kind, name = kw, kw
        rec = contract_at(start)
        fn_rows.append(analyze_function(
            clean, brace_close, paren_close, start, hdr_end, fc, name, kind, rec,
            state_vars_for(rec), solc_below))

    if not fn_rows and src.strip():
        warnings.append(("no_functions_detected", "no function with a body was found"))

    # ---- aggregate function features
    out = {}
    if fn_rows:
        mat = np.array([[r[n] for n in FN_NAMES] for r in fn_rows], dtype=np.float64)
        sums, maxs, means = mat.sum(0), mat.max(0), mat.mean(0)
    else:
        sums = maxs = means = np.zeros(len(FN_NAMES))
    for i, n in enumerate(FN_NAMES):
        out[f"fn_{n}_sum"], out[f"fn_{n}_max"], out[f"fn_{n}_mean"] = sums[i], maxs[i], means[i]
    fsum = dict(zip(FN_NAMES, sums))

    # ---- contract-level features
    words = Counter(WORD_RE.findall(clean))
    wget = words.get
    ct["c_n_contracts"] = sum(1 for c in contracts if c["kind"] == "contract")
    ct["c_n_abstract_contracts"] = sum(1 for c in contracts if c["abstract"])
    ct["c_n_libraries"] = sum(1 for c in contracts if c["kind"] == "library")
    ct["c_n_interfaces"] = sum(1 for c in contracts if c["kind"] == "interface")
    ct["c_n_functions"] = int(fsum["is_function"])
    ct["c_n_constructors"] = int(fsum["is_constructor"])
    ct["c_n_modifiers_defined"] = int(fsum["is_modifier"])
    ct["c_n_fallback"] = int(fsum["is_fallback"])
    ct["c_n_receive"] = int(fsum["is_receive"])
    ct["c_n_bodyless_function_decls"] = n_bodyless
    ct["c_n_state_vars_est"] = sum(c["n_state_vars"] for c in contracts)
    ct["c_n_events"], ct["c_n_structs"], ct["c_n_enums"] = wget("event", 0), wget("struct", 0), wget("enum", 0)
    ct["c_n_mappings"], ct["c_n_using_for"], ct["c_n_imports"] = wget("mapping", 0), wget("using", 0), wget("import", 0)
    ct["c_n_inheritance"] = sum(len(c["bases"]) for c in contracts)
    ct["c_max_inheritance"] = max((len(c["bases"]) for c in contracts), default=0)
    ct["c_uses_safemath"] = int(wget("SafeMath", 0) > 0)
    ct["c_n_assembly"] = len(ASSEMBLY_RE.findall(clean))
    ct["c_uses_assembly"] = int(ct["c_n_assembly"] > 0)
    ct["c_src_chars"] = len(src)
    ct["c_src_lines"] = src.count("\n") + 1 if src else 0
    ct["c_code_lines"] = sum(1 for ln in clean.split("\n") if ln.strip())
    ct["c_comment_ratio"] = comment_chars / len(src) if src else 0.0
    ct["c_no_functions_detected"] = int(not fn_rows)

    # whole-file counts (independent of function parsing)
    ext_total = low_total = eth_out_file = 0
    if any(wget(w) for w in ("call", "delegatecall", "staticcall", "callcode", "send", "transfer")):
        for m in EXT_RE.finditer(clean):
            k = m.group(1)
            if k in ("send", "transfer"):
                j = next_nonspace(clean, m.end())
                if j >= len(clean) or clean[j] != "(":
                    continue
                if k == "send":
                    eth_out_file += 1
                else:
                    c = paren_close.get(j)
                    if c is not None and clean[j + 1:c].strip() and top_level_commas(clean[j + 1:c]) == 0:
                        eth_out_file += 1
            ext_total += 1
            if k != "transfer":
                low_total += 1
    value_file = len(VALUE_RE.findall(clean)) if wget("value") else 0
    sd_file = len(SELFDESTRUCT_RE.findall(clean))
    ct["c_total_ext_interactions"] = ext_total
    ct["c_total_low_level_calls"] = low_total
    ct["c_total_checks"] = (len(REQUIRE_CALL_RE.findall(clean)) + len(ASSERT_CALL_RE.findall(clean))
                             + len(REVERT_RE.findall(clean)) + len(re.findall(r"\bthrow\b", clean)))
    arith_file = 0
    for m in OP_RE.finditer(clean):
        t = m.group()
        if t in ("+", "-"):
            j = m.start() - 1
            while j >= 0 and clean[j] in " \t\r\n":
                j -= 1
            if j < 0 or not (clean[j].isalnum() or clean[j] in "_$)]"):
                continue
        if t in OTHER_COMPOUND:
            continue
        arith_file += 1
    ct["c_total_arith_ops"] = arith_file
    ct["c_total_payable_functions"] = int(fsum["is_payable"])
    ct["c_total_external_functions"] = int(fsum["vis_external"])
    ct["c_total_public_functions"] = int(fsum["eff_public"])
    payable_entry = int(fsum["is_payable"] > 0)
    ether_out = int(eth_out_file + value_file + sd_file > 0)
    ct["c_has_payable_entry"] = payable_entry
    ct["c_has_ether_out"] = ether_out
    ct["c_payable_no_ether_out_est"] = int(bool(payable_entry and not ether_out))
    ct["c_arith_unprotected_est"] = int(bool(
        ct["solc_below_0_8"] == 1 and ct["c_total_arith_ops"] > 0
        and not ct["c_uses_safemath"] and fsum["n_safemath_like_calls"] == 0))
    ct["file_n_tx_origin"] = len(TX_ORIGIN_RE.findall(clean)) if wget("tx") else 0
    ct["file_n_delegatecall"] = wget("delegatecall", 0)
    ct["file_n_selfdestruct"] = sd_file
    ct["file_n_block_timestamp"] = len(re.findall(r"\bnow\b", clean)) + len(
        re.findall(r"\bblock\s*\.\s*timestamp\b", clean))
    ct["file_n_blockhash"] = len(BLOCKHASH_RE.findall(clean))
    ct["file_n_ecrecover"] = wget("ecrecover", 0)

    out.update(ct)
    return out, warnings


# ----------------------------------------------------------------------------
# Multiprocessing worker (never raises)
# ----------------------------------------------------------------------------
_TIMEOUT = 0


class _FileTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FileTimeout()


def _init_worker(timeout):
    global _TIMEOUT
    _TIMEOUT = timeout


def process_one(job):
    source_hash, source = job
    errors, feats, ok = [], None, 1
    use_alarm = _TIMEOUT > 0 and hasattr(signal, "SIGALRM")
    try:
        if use_alarm:
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(_TIMEOUT))
        feats, warns = extract_features(source)
        for kind, msg in warns:
            errors.append({"source_hash": source_hash, "severity": "warning",
                           "error_type": kind, "message": msg})
    except _FileTimeout:
        ok = 0
        errors.append({"source_hash": source_hash, "severity": "error",
                       "error_type": "timeout", "message": f"exceeded {_TIMEOUT}s"})
    except Exception as exc:                                   # noqa: BLE001 - one file must never stop the run
        ok = 0
        tb = traceback.format_exc(limit=3).replace("\n", " | ")
        errors.append({"source_hash": source_hash, "severity": "error",
                       "error_type": type(exc).__name__, "message": f"{exc} :: {tb}"[:1000]})
    finally:
        if use_alarm:
            signal.alarm(0)
    return source_hash, feats, ok, errors


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
def parse_labels(x) -> list:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [int(i) for i in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return [int(i) for i in x]
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, float):
        return [] if math.isnan(x) else [int(x)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            return parse_labels(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return [int(t) for t in re.findall(r"-?\d+", s)]
    return []


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def feature_column_names():
    cols = []
    for n in FN_NAMES:
        cols += [f"fn_{n}_{a}" for a in AGGS]
    return cols + CT_NAMES


def build_dictionary_rows():
    rows = []
    for n in FN_NAMES:
        rel, desc = FN_META[n]
        for a in AGGS:
            rows.append({"feature": f"fn_{n}_{a}", "level": "function->contract aggregate",
                         "aggregation": a, "reliability": rel, "description": f"{desc} [{a} over functions]"})
    for n in CT_NAMES:
        rel, desc = CT_META[n]
        rows.append({"feature": n, "level": "contract/file", "aggregation": "",
                     "reliability": rel, "description": desc})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Compiler-free Solidity feature extraction")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel processes (start with 1 or 2 in Colab)")
    ap.add_argument("--timeout", type=int, default=60,
                    help="per-file time limit in seconds (0 = off; Unix only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N rows; useful for testing")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--restart", action="store_true",
                    help="delete previous batch checkpoints and start over")
    args = ap.parse_args()

    if args.workers < 1 or args.batch_size < 1:
        ap.error("--workers and --batch-size must be >= 1")

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    parts_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(parts_dir, exist_ok=True)
    if args.restart:
        for name in os.listdir(parts_dir):
            if name.endswith(".parquet") or name.endswith(".csv"):
                os.remove(os.path.join(parts_dir, name))
        print("[info] previous checkpoints cleared")

    df = pd.read_parquet(args.input)
    required = ["source_hash", "source_code", "labels", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[fatal] input is missing required columns: {missing}")
    if df["source_hash"].duplicated().any():
        dup = int(df["source_hash"].duplicated().sum())
        print(f"[warn] dropping {dup} duplicate source_hash rows (keeping first)")
        df = df.drop_duplicates("source_hash", keep="first")
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()

    # A completed batch is an atomic parquet file. If Colab stops mid-batch,
    # that batch is rerun; completed batches are skipped on the next execution.
    part_files = sorted(os.path.join(parts_dir, f) for f in os.listdir(parts_dir)
                        if f.startswith("batch_") and f.endswith(".parquet"))
    completed = set()
    for path in part_files:
        try:
            part = pd.read_parquet(path, columns=["source_hash"])
            completed.update(part["source_hash"].astype(str).tolist())
        except Exception as exc:
            print(f"[warn] ignoring unreadable checkpoint {path}: {exc}")

    todo = df[~df["source_hash"].astype(str).isin(completed)].copy()
    print(f"[info] total={len(df)} | already complete={len(completed)} | remaining={len(todo)}")
    print(f"[info] workers={args.workers} | batch_size={args.batch_size} | timeout={args.timeout}s")

    feat_cols = feature_column_names()
    err_cols = ["source_hash", "severity", "error_type", "message"]
    batch_no = 0
    # Count only checkpointed hashes that belong to the current selected input.
    completed_count = len(df) - len(todo)
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for start_idx in range(0, len(todo), args.batch_size):
        batch = todo.iloc[start_idx:start_idx + args.batch_size].copy()
        jobs = list(zip(batch["source_hash"].astype(str).tolist(), batch["source_code"].tolist()))
        if args.workers > 1:
            with Pool(args.workers, initializer=_init_worker, initargs=(args.timeout,)) as pool:
                results = list(pool.imap(process_one, jobs, chunksize=4))
        else:
            _init_worker(args.timeout)
            results = [process_one(job) for job in jobs]

        zero = {c: 0.0 for c in feat_cols}
        rows, ok_flags, errors = [], [], []
        for (h, feats, ok, errs), expected in zip(results, batch["source_hash"].astype(str)):
            if h != expected:
                raise RuntimeError("Result order mismatch; batch was not saved")
            rows.append({c: feats.get(c, 0.0) for c in feat_cols} if feats is not None else dict(zero))
            ok_flags.append(ok)
            errors.extend(errs)

        feat_df = pd.DataFrame(rows, columns=feat_cols).astype(np.float32)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        label_lists = [parse_labels(x) for x in batch["labels"]]
        unknown = sorted({i for ls in label_lists for i in ls if i not in LABEL_NAMES})
        if unknown:
            print(f"[warn] unknown label ids: {unknown}")
        targets = pd.DataFrame({target_col(i): [int(i in ls) for ls in label_lists]
                                for i in TARGET_IDS})
        meta = pd.DataFrame({"source_hash": batch["source_hash"].values,
                             "split": batch["split"].values,
                             "labels": batch["labels"].values})
        for col in ("duplicate_count", "original_splits"):
            if col in batch.columns:
                meta[col] = batch[col].values
        meta["extraction_ok"] = np.asarray(ok_flags, dtype=np.int8)
        out = pd.concat([meta.reset_index(drop=True), targets.reset_index(drop=True),
                         feat_df.reset_index(drop=True)], axis=1)

        assert out["source_hash"].is_unique
        assert (out["split"].values == batch["split"].values).all()
        assert all(np.issubdtype(feat_df[c].dtype, np.number) for c in feat_cols)
        assert np.isfinite(feat_df.to_numpy()).all()

        batch_no += 1
        part_path = os.path.join(parts_dir, f"batch_{int(time.time())}_{batch_no:05d}.parquet")
        # Write to temporary path then atomically rename to avoid partial checkpoints.
        tmp_path = part_path + ".tmp"
        out.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, part_path)
        if errors:
            pd.DataFrame(errors, columns=err_cols).to_csv(
                part_path.replace(".parquet", "_errors.csv"), index=False)
        completed_count += len(out)
        percent = 100.0 * completed_count / max(len(df), 1)
        remaining = max(len(df) - completed_count, 0)
        current_batch = (start_idx // args.batch_size) + 1
        print(
            f"[progress] batch {current_batch}/{total_batches} | "
            f"saved this batch={len(out)} | "
            f"TOTAL parsed={completed_count:,}/{len(df):,} ({percent:.1f}%) | "
            f"remaining={remaining:,} | "
            f"failures in this batch={int((out['extraction_ok'] == 0).sum())}",
            flush=True
        )

    # Merge all checkpoint batches into final deliverables.
    part_files = sorted(os.path.join(parts_dir, f) for f in os.listdir(parts_dir)
                        if f.startswith("batch_") and f.endswith(".parquet"))
    if not part_files:
        print("[info] no new work and no checkpoints found")
        return
    pieces = [pd.read_parquet(path) for path in part_files]
    final = pd.concat(pieces, ignore_index=True).drop_duplicates("source_hash", keep="last")
    # Keep original input order and verify that every selected source appears once.
    order = {str(h): i for i, h in enumerate(df["source_hash"].astype(str))}
    final["_order"] = final["source_hash"].astype(str).map(order)
    final = final.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(final) != len(df):
        missing_hashes = set(df["source_hash"].astype(str)) - set(final["source_hash"].astype(str))
        raise RuntimeError(f"Final output incomplete: {len(final)}/{len(df)} rows; missing={len(missing_hashes)}")
    assert final["source_hash"].is_unique
    assert (final["split"].values == df["split"].values).all()
    leaked = [c for c in feat_cols if c.startswith("target_") or c in
              {"source_hash", "split", "labels", "duplicate_count", "original_splits"}]
    assert not leaked, f"metadata/target leakage into features: {leaked}"

    out_path = os.path.join(args.outdir, "contract_features.parquet")
    final.to_parquet(out_path, index=False)
    error_parts = sorted(os.path.join(parts_dir, f) for f in os.listdir(parts_dir)
                         if f.startswith("batch_") and f.endswith("_errors.csv"))
    error_frames = [pd.read_csv(x) for x in error_parts]
    all_errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=err_cols)
    all_errors.to_csv(os.path.join(args.outdir, "feature_extraction_errors.csv"), index=False)
    pd.DataFrame(build_dictionary_rows()).to_csv(
        os.path.join(args.outdir, "feature_dictionary.csv"), index=False)
    with open(os.path.join(args.outdir, "feature_columns.json"), "w") as fh:
        json.dump({"feature_columns": feat_cols, "target_columns": TARGET_COLS,
                   "metadata_columns": [c for c in final.columns
                                        if c in ("source_hash", "split", "labels", "duplicate_count",
                                                 "original_splits", "extraction_ok")]}, fh, indent=2)

    print(f"\n[done] {len(final)} rows x {final.shape[1]} columns -> {out_path}")
    print(f"[done] features={len(feat_cols)} | targets={len(TARGET_COLS)}")
    print(f"[done] extraction failures={int((final['extraction_ok'] == 0).sum())}")
    if len(all_errors):
        print("[info] error/warning counts:")
        print(all_errors.groupby(["severity", "error_type"]).size().to_string())
    print(f"[info] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()