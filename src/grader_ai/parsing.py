from dataclasses import dataclass
from typing import Final

from pylatexenc.latexwalker import (
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexWalker,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec, MacroStandardArgsParser

_PROBLEM_MACROS: Final = ["problemTF", "problemMC", "problemPS", "problemAI"]


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
    context_db = get_default_latex_context_db()
    context_db.add_context_category(
        category="grader_ai",
        macros=[
            MacroSpec("problemTF", MacroStandardArgsParser("{")),
            MacroSpec("problemMC", MacroStandardArgsParser("{")),
            MacroSpec("problemPS", MacroStandardArgsParser("{")),
            MacroSpec("problemAI", MacroStandardArgsParser("{")),
            MacroSpec("answer", MacroStandardArgsParser("{")),
            MacroSpec("solution", MacroStandardArgsParser("{")),
        ],
    )

    reference_walker = LatexWalker(reference, latex_context=context_db)

    problems = _extract_problems(reference_walker)
    answers = _extract_by_macro(reference_walker, macro="answer")

    submission_walker = LatexWalker(submission, latex_context=context_db)

    responses = _extract_by_macro(submission_walker, macro="solution")

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


def _extract_problems(latex_walker: LatexWalker) -> list[_Problem]:
    credits_by_macro = _extract_credits_by_macro(latex_walker, macros=_PROBLEM_MACROS)

    results: list[_Problem] = []

    nodes = _get_document_nodes(latex_walker)

    for node in nodes:
        if not isinstance(node, LatexMacroNode):
            continue

        if node.macroname not in _PROBLEM_MACROS:
            continue

        content = "".join(
            subnode.latex_verbatim() for subnode in node.nodeargd.argnlist[0].nodelist
        )

        results.append(
            _Problem(content=content, credits=credits_by_macro[node.macroname])
        )

    return results


def _extract_credits_by_macro(
    latex_walker: LatexWalker, *, macros: list[str]
) -> dict[str, int]:
    results: dict[str, int] = {}

    nodes, _, _ = latex_walker.get_latex_nodes()

    for node in nodes:
        if not isinstance(node, LatexMacroNode):
            continue

        if node.macroname != "newcommand":
            continue

        args = [arg for arg in node.nodeargd.argnlist if arg is not None]

        names = [n for n in args[0].nodelist if isinstance(n, LatexMacroNode)]
        if len(names) != 1:
            continue

        macroname = names[0].macroname
        if macroname not in macros:
            continue

        results[macroname] = int(args[2].nodelist[0].chars)

    return results


def _extract_by_macro(latex_walker: LatexWalker, *, macro: str) -> list[str]:
    results: list[str] = []

    nodes = _get_document_nodes(latex_walker)

    for node in nodes:
        if not isinstance(node, LatexMacroNode):
            continue

        if node.macroname != macro:
            continue

        content = "".join(
            subnode.latex_verbatim() for subnode in node.nodeargd.argnlist[0].nodelist
        )

        results.append(content)

    return results


def _get_document_nodes(latex_walker: LatexWalker) -> list:
    nodes, _, _ = latex_walker.get_latex_nodes()

    for node in nodes:
        if not isinstance(node, LatexEnvironmentNode):
            continue

        if node.environmentname != "document":
            continue

        return node.nodelist

    return []
