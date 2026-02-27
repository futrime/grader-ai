from dataclasses import dataclass
import re
from typing import Any

from pylatexenc.latexwalker import LatexGroupNode, LatexMacroNode, LatexWalker
from pylatexenc.macrospec import LatexContextDb, MacroSpec, MacroStandardArgsParser

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


def parse(reference: str, submission: str) -> list[ParseResult]:
    problems = _extract_problems(reference)
    credits_by_macro = _extract_problem_credits(reference)
    answers = _extract_macro_arguments(reference, "answer")
    responses = _extract_macro_arguments(submission, "solution")
    return [
        ParseResult(
            problem=problem,
            credits=credits_by_macro.get(problem_macro, 0),
            answer=answer,
            response=response,
        )
        for (problem_macro, problem), answer, response in zip(
            problems, answers, responses
        )
    ]


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
    _walk(nodes, macro_name, extracted)
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
    _walk_problems(nodes, set(_PROBLEM_MACROS), extracted)
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


def _walk(nodes: list[Any], macro_name: str, extracted: list[str]) -> None:
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            if (
                node.macroname == macro_name
                and node.nodeargd
                and node.nodeargd.argnlist
            ):
                first_arg = node.nodeargd.argnlist[0]
                if isinstance(first_arg, LatexGroupNode):
                    content = first_arg.latex_verbatim()[1:-1]
                    if content:
                        extracted.append(content)
                elif first_arg is not None:
                    content = first_arg.latex_verbatim()
                    if content:
                        extracted.append(content)

            if node.macroname in _DEFINITION_MACROS:
                continue

            if node.nodeargd:
                for arg in node.nodeargd.argnlist:
                    if isinstance(arg, LatexGroupNode):
                        _walk(arg.nodelist, macro_name, extracted)

        nodelist = getattr(node, "nodelist", None)
        if nodelist:
            _walk(nodelist, macro_name, extracted)


def _walk_problems(
    nodes: list[Any], macro_names: set[str], extracted: list[tuple[str, str]]
) -> None:
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            if (
                node.macroname in macro_names
                and node.nodeargd
                and node.nodeargd.argnlist
            ):
                first_arg = node.nodeargd.argnlist[0]
                if isinstance(first_arg, LatexGroupNode):
                    content = first_arg.latex_verbatim()[1:-1]
                    if content:
                        extracted.append((node.macroname, content))
                elif first_arg is not None:
                    content = first_arg.latex_verbatim()
                    if content:
                        extracted.append((node.macroname, content))

            if node.macroname in _DEFINITION_MACROS:
                continue

            if node.nodeargd:
                for arg in node.nodeargd.argnlist:
                    if isinstance(arg, LatexGroupNode):
                        _walk_problems(arg.nodelist, macro_names, extracted)

        nodelist = getattr(node, "nodelist", None)
        if nodelist:
            _walk_problems(nodelist, macro_names, extracted)
