from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import re
from typing import Any, TypeVar

from pylatexenc.latexwalker import LatexGroupNode, LatexMacroNode, LatexWalker
from pylatexenc.macrospec import LatexContextDb, MacroSpec, MacroStandardArgsParser

logger = logging.getLogger(__name__)

_DEFINITION_MACROS = {
    "newcommand",
    "renewcommand",
    "providecommand",
    "DeclareRobustCommand",
}

_PROBLEM_MACROS = ("problemTF", "problemMC", "problemPS", "problemAI")


@dataclass(frozen=True)
class ParseResult:
    problem: str
    credits: int
    answer: str
    response: str


@dataclass(frozen=True)
class ParseOutcome:
    """Result of parsing a reference/submission pair."""

    results: list[ParseResult]
    warnings: list[str] = field(default_factory=list)


def parse(reference: str, submission: str) -> ParseOutcome:
    """Parse reference and submission LaTeX, returning results and warnings.

    Warnings are emitted when:
    - No problems are found in the reference.
    - No responses are found in the submission.
    - The number of responses does not match the number of problems.
    - A problem has no credit definition.
    - A problem has no matching answer in the reference.
    """
    warnings: list[str] = []

    problems = _extract_problems(reference)
    credits_by_macro = _extract_problem_credits(reference)
    answers = _extract_macro_arguments(reference, "answer")
    responses = _extract_macro_arguments(submission, "solution")

    if not problems:
        msg = "No problems found in the reference."
        warnings.append(msg)
        logger.warning(msg)

    if not responses:
        msg = "No responses (\\solution{}) found in the submission."
        warnings.append(msg)
        logger.warning(msg)

    if problems and responses and len(responses) != len(problems):
        msg = (
            f"Response count ({len(responses)}) does not match "
            f"problem count ({len(problems)}); "
            f"only the first {min(len(problems), len(responses))} "
            f"will be graded."
        )
        warnings.append(msg)
        logger.warning(msg)

    matched_count = min(len(problems), len(responses))
    results: list[ParseResult] = []
    for index in range(matched_count):
        macro_name, problem_text = problems[index]

        credits = credits_by_macro.get(macro_name, 0)
        if credits == 0:
            msg = (
                f"Problem {index + 1} (\\{macro_name}): "
                f"no credit definition found; defaulting to 0."
            )
            warnings.append(msg)
            logger.warning(msg)

        answer = answers[index] if index < len(answers) else ""
        if not answer:
            msg = (
                f"Problem {index + 1} (\\{macro_name}): "
                f"no matching \\answer{{}} in the reference."
            )
            warnings.append(msg)
            logger.warning(msg)

        results.append(
            ParseResult(
                problem=problem_text,
                credits=credits,
                answer=answer,
                response=responses[index],
            )
        )

    return ParseOutcome(results=results, warnings=warnings)


_T = TypeVar("_T")


def _walk_macros(
    nodes: list[Any],
    match: set[str],
    transform: Callable[[LatexMacroNode, str], _T],
    extracted: list[_T],
) -> None:
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            if node.macroname in match and node.nodeargd and node.nodeargd.argnlist:
                first_arg = node.nodeargd.argnlist[0]
                if isinstance(first_arg, LatexGroupNode):
                    content = first_arg.latex_verbatim()[1:-1]
                elif first_arg is not None:
                    content = first_arg.latex_verbatim()
                else:
                    content = ""
                if content:
                    extracted.append(transform(node, content))

            if node.macroname in _DEFINITION_MACROS:
                continue

            if node.nodeargd:
                for arg in node.nodeargd.argnlist:
                    if isinstance(arg, LatexGroupNode):
                        _walk_macros(arg.nodelist, match, transform, extracted)

        nodelist = getattr(node, "nodelist", None)
        if nodelist:
            _walk_macros(nodelist, match, transform, extracted)


def _extract_macro_arguments(latex: str, macro_name: str) -> list[str]:
    context = LatexContextDb()
    one_braced_arg = MacroStandardArgsParser("{")
    context.add_context_category(
        "grader_ai",
        macros=[
            MacroSpec("answer", one_braced_arg),
            MacroSpec("solution", one_braced_arg),
        ],
    )
    nodes, _, _ = LatexWalker(latex, latex_context=context).get_latex_nodes(pos=0)
    extracted: list[str] = []
    _walk_macros(nodes, {macro_name}, lambda _n, c: c, extracted)
    return extracted


def _extract_problems(latex: str) -> list[tuple[str, str]]:
    context = LatexContextDb()
    one_braced_arg = MacroStandardArgsParser("{")
    context.add_context_category(
        "grader_ai_problems",
        macros=[MacroSpec(name, one_braced_arg) for name in _PROBLEM_MACROS],
    )
    nodes, _, _ = LatexWalker(latex, latex_context=context).get_latex_nodes(pos=0)
    extracted: list[tuple[str, str]] = []
    _walk_macros(nodes, set(_PROBLEM_MACROS), lambda n, c: (n.macroname, c), extracted)
    return extracted


def _extract_problem_credits(latex: str) -> dict[str, int]:
    credits: dict[str, int] = {}
    for macro_name in _PROBLEM_MACROS:
        definition_match = re.search(
            rf"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\s*\{{\\{macro_name}\}}",
            latex,
        )
        if not definition_match:
            continue

        following = latex[definition_match.end() :]
        points_match = re.search(r"\((\d+)\s*pts\)", following)
        if points_match:
            credits[macro_name] = int(points_match.group(1))

    return credits
