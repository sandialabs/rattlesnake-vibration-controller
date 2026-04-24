#!/usr/bin/env python3
"""
Generates the root index.html Project Dashboard for GitHub Pages.
"""

import argparse
import os


def generate_dashboard_html(github_repo: str) -> str:
    """
    Generate the HTML content for the Project Dashboard using a two-column layout.

    Args:
        github_repo: GitHub repository name (owner/repo)

    Returns:
        HTML string
    """
    # Using Sandia Blue as the primary accent to match Jupyter Book theme
    sandia_blue = "#005376"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rattlesnake | Project Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
          rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .text-sandia {{ color: {sandia_blue}; }}
        .bg-sandia {{ background-color: {sandia_blue}; }}
        .border-sandia {{ border-color: {sandia_blue}; }}
        .hover-card:hover {{ transform: translateY(-2px); transition: all 0.2s ease; }}
    </style>
    </head>
    <body class="bg-slate-50 text-slate-800 antialiased min-h-screen">

    <nav class="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div class="flex items-center gap-2">
                <span class="text-2xl">🐍</span>
                <span class="text-xl font-bold tracking-tight text-sandia">Rattlesnake</span>
            </div>
            <a href="https://github.com/{github_repo}"
               class="text-sm font-medium hover:text-sandia transition">GitHub Repository</a>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <header class="mb-12 text-center">
            <h1 class="text-4xl font-extrabold tracking-tight text-slate-900 mb-4">
                <b>Project Dashboard</b></h1>
            <p class="text-lg text-slate-600 max-w-2xl mx-auto">
                Access documentation and quality reports for the Rattlesnake Vibration Controller.
            </p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">

            <section class="space-y-6">
                <div class="flex items-center justify-between border-b-2 border-blue-200 pb-2">
                    <h2 class="text-2xl font-bold flex items-center gap-2">
                        🚀 Released
                    </h2>
                    <a href="https://github.com/{github_repo}/tree/main"
                       class="text-xs font-semibold bg-blue-100 text-blue-700
                              px-2 py-1 rounded hover:bg-slate-300">main branch</a>
                </div>

                <div class="space-y-4">
                    <a href="main/book/jupyter/index.html"
                       class="hover-card block p-5 bg-white rounded-xl shadow-sm border
                              border-slate-200 hover:border-sandia group">
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-bold text-lg group-hover:text-sandia">
                                    User's Manual</h3>
                                <p class="text-sm text-slate-500 mt-1">
                                    Stable documentation for end-users.</p>
                            </div>
                            <span class="text-xs font-bold uppercase tracking-wider
                                         bg-blue-100 text-blue-700 px-2 py-1 rounded">
                                Stable</span>
                        </div>
                    </a>

                    <div class="grid grid-cols-2 gap-4">
                        <a href="main/reports/lint/index.html"
                           class="hover-card p-4 bg-white rounded-xl shadow-sm border
                                  border-slate-200 hover:border-sandia">
                            <span class="block text-xs font-bold text-slate-400
                                         uppercase mb-2">Code Quality</span>
                            <img src="main/badges/lint.svg" alt="Lint Score" class="h-5">
                        </a>
                        <a href="main/reports/coverage/index.html"
                           class="hover-card p-4 bg-white rounded-xl shadow-sm border
                                  border-slate-200 hover:border-sandia">
                            <span class="block text-xs font-bold text-slate-400
                                         uppercase mb-2">Test Coverage</span>
                            <img src="main/badges/coverage.svg" alt="Coverage" class="h-5">
                        </a>
                    </div>
                </div>
            </section>

            <section class="space-y-6">
                <div class="flex items-center justify-between border-b-2 border-orange-200 pb-2">
                    <h2 class="text-2xl font-bold flex items-center gap-2">
                        🛠️ Development
                    </h2>
                    <a href="https://github.com/{github_repo}/tree/dev"
                       class="text-xs font-semibold bg-orange-100 text-orange-700
                              px-2 py-1 rounded hover:bg-orange-200">dev branch</a>
                </div>

                <div class="space-y-4">
                    <a href="dev/book/jupyter/index.html"
                       class="hover-card block p-5 bg-white rounded-xl shadow-sm border
                              border-slate-200 hover:border-orange-500 group">
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-bold text-lg group-hover:text-orange-600">
                                    User's Manual</h3>
                                <p class="text-sm text-slate-500 mt-1">
                                    Development documentation.</p>
                            </div>
                            <span class="text-xs font-bold uppercase tracking-wider
                                         bg-orange-100 text-orange-700 px-2 py-1 rounded">
                                Latest</span>
                        </div>
                    </a>

                    <div class="grid grid-cols-2 gap-4">
                        <a href="dev/reports/lint/index.html"
                           class="hover-card p-4 bg-white rounded-xl shadow-sm border
                                  border-slate-200 hover:border-orange-500">
                            <span class="block text-xs font-bold text-slate-400
                                         uppercase mb-2">Code Quality</span>
                            <img src="dev/badges/lint.svg" alt="Lint Score" class="h-5">
                        </a>
                        <a href="dev/reports/coverage/index.html"
                           class="hover-card p-4 bg-white rounded-xl shadow-sm border
                                  border-slate-200 hover:border-orange-500">
                            <span class="block text-xs font-bold text-slate-400
                                         uppercase mb-2">Test Coverage</span>
                            <img src="dev/badges/coverage.svg" alt="Coverage" class="h-5">
                        </a>
                    </div>
                </div>
            </section>

        </div>
    </main>

    <footer class="mt-20 py-10 border-t border-slate-200 bg-white">
        <div class="max-w-7xl mx-auto px-4 text-center">
            <p class="text-sm text-slate-400">
                &copy; 2026 Sandia National Laboratories | Released under GPL-3.0
            </p>
        </div>
    </footer>

</body>
</html>"""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Project Dashboard index.html")
    parser.add_argument("--github_repo", required=True, help="GitHub repository (owner/repo)")
    parser.add_argument("--output_file", required=True, help="Output HTML file path")
    args = parser.parse_args()

    html = generate_dashboard_html(args.github_repo)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Project Dashboard generated: {args.output_file}")


if __name__ == "__main__":
    main()
