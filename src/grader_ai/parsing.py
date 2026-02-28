import re
from dataclasses import dataclass
from typing import Any, Final

from pylatexenc.latexwalker import (
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
)
from pylatexenc.macrospec import LatexContextDb, MacroSpec, MacroStandardArgsParser

_PROBLEM_MACRO_PATTERN: Final = re.compile(r"problem[A-Z]+")


@dataclass(frozen=True)
class ParseResult:
    problem: str
    credits: int
    answer: str
    response: str


@dataclass(frozen=True)
class _Problem:
    content: str
    credits: int


def parse(reference: str, submission: str) -> list[ParseResult]:
    problems = _extract_problems(reference)
    answers = _extract_macro_arguments(reference, "answer")
    responses = _extract_macro_arguments(submission, "solution")

    results: list[ParseResult] = []
    for index, problem in enumerate(problems):
        answer = answers[index]

        response = responses[index] if index < len(responses) else ""

        results.append(
            ParseResult(
                problem=problem.content,
                credits=problem.credits,
                answer=answer,
                response=response,
            )
        )

    return results


def _document_children(
    latex: str, *, context: LatexContextDb | None = None
) -> list[Any]:
    nodes, _, _ = LatexWalker(latex, latex_context=context).get_latex_nodes(pos=0)
    document = next(
        node
        for node in nodes
        if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document"
    )
    return document.nodelist


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
    nodes = _document_children(latex, context=context)
    extracted: list[str] = []
    for node in nodes:
        if not isinstance(node, LatexMacroNode) or node.macroname != macro_name:
            continue
        extracted.append(_first_argument_text(node))
    return extracted


def _extract_problems(latex: str) -> list[_Problem]:
    macro_names = _extract_problem_macro_names(latex)
    credits_by_macro: dict[str, int] = {}
    for macro_name in macro_names:
        definition_match = re.search(
            rf"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\s*\{{\\{macro_name}\}}",
            latex,
        )
        if not definition_match:
            continue

        following = latex[definition_match.end() :]
        points_match = re.search(r"\((\d+)\s*pts\)", following)
        if points_match:
            credits_by_macro[macro_name] = int(points_match.group(1))

    context = LatexContextDb()
    one_braced_arg = MacroStandardArgsParser("{")
    context.add_context_category(
        "grader_ai_problems",
        macros=[MacroSpec(name, one_braced_arg) for name in macro_names],
    )
    nodes = _document_children(latex, context=context)
    extracted: list[_Problem] = []
    for node in nodes:
        if (
            not isinstance(node, LatexMacroNode)
            or _PROBLEM_MACRO_PATTERN.fullmatch(node.macroname) is None
        ):
            continue
        extracted.append(
            _Problem(
                content=_first_argument_text(node),
                credits=credits_by_macro.get(node.macroname, 0),
            )
        )
    return extracted


def _first_argument_text(node: LatexMacroNode) -> str:
    first_arg = node.nodeargd.argnlist[0]
    if isinstance(first_arg, LatexGroupNode):
        return first_arg.latex_verbatim()[1:-1]
    return first_arg.latex_verbatim()


def _extract_problem_macro_names(latex: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for macro_name in _PROBLEM_MACRO_PATTERN.findall(latex):
        if macro_name in seen:
            continue
        seen.add(macro_name)
        ordered.append(macro_name)
    return ordered
