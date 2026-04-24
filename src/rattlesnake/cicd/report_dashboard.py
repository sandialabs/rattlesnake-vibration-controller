#!/usr/bin/env python3
"""
Generates the root index.html Project Dashboard for GitHub Pages.
"""

import argparse
import os

from rattlesnake.cicd.utilities import (
    ReportMetadata,
    add_common_args,
    extend_timestamp,
    report_metadata_creation,
)


def generate_dashboard_html(metadata: ReportMetadata) -> str:
    """
    Generate the HTML content for the Project Dashboard using a two-column layout.

    Args:
        metadata: CI/CD metadata

    Returns:
        HTML string
    """
    timestamp_ext = extend_timestamp(metadata.timestamp)

    # Pre-calculate GitHub URLs to stay under line length limits
    github_url = f"https://github.com/{metadata.github_repo}"
    run_url = f"{github_url}/actions/runs/{metadata.run_id}"
    branch_url = f"{github_url}/tree/{metadata.ref_name}"
    commit_url = f"{github_url}/commit/{metadata.github_sha}"

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
        .hover-card:hover {{ transform: translateY(-2px); transition: all 0.2s ease; }}
    </style>
    </head>
    <body class="bg-slate-50 text-slate-800 antialiased min-h-screen">
<nav class="bg-white border-b border-slate-200 sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <div class="flex items-center gap-2">
            <span class="text-xl font-bold tracking-tight text-blue-600">Rattlesnake</span>
        </div>

            <a href="{github_url}"
               class="text-sm font-medium hover:text-blue-600 transition">GitHub Repository</a>
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
                    <a href="{github_url}/tree/main"
                       class="text-xs font-semibold bg-blue-100 text-blue-700
                              px-2 py-1 rounded hover:bg-slate-300">main branch</a>
                </div>

                <div class="space-y-4">
                    <a href="main/book/jupyter/index.html"
                       class="hover-card block p-5 bg-white rounded-xl shadow-sm border
                              border-slate-200 hover:border-blue-600 group">
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-bold text-lg group-hover:text-blue-600">
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
                                  border-slate-200 hover:border-blue-600">
                            <span class="block text-xs font-bold text-slate-400
                                         uppercase mb-2">Code Quality</span>
                            <img src="main/badges/lint.svg" alt="Lint Score" class="h-5">
                        </a>
                        <a href="main/reports/coverage/index.html"
                           class="hover-card p-4 bg-white rounded-xl shadow-sm border
                                  border-slate-200 hover:border-blue-600">
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
                    <a href="{github_url}/tree/dev"
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

        <section class="mt-16 bg-white p-8 rounded-xl shadow-sm border border-slate-200">
            <h2 class="text-xl font-bold mb-6 border-b pb-2">Build Metadata</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-y-4 gap-x-12 text-sm text-slate-600">
                <div>
                    <span class="font-semibold">Generated:</span> {timestamp_ext}
                </div>
                <div>
                    <span class="font-semibold">Run ID:</span>
                    <a href="{run_url}" class="text-blue-600 hover:underline">{metadata.run_id}</a>
                </div>
                <div>
                    <span class="font-semibold">Branch:</span>
                    <a href="{branch_url}" class="text-blue-600 hover:underline">{metadata.ref_name}</a>
                </div>
                <div>
                    <span class="font-semibold">Commit:</span>
                    <a href="{commit_url}" class="text-blue-600 hover:underline">{metadata.github_sha[:7]}</a>
                </div>
                <div>
                    <span class="font-semibold">Repository:</span>
                    <a href="{github_url}" class="text-blue-600 hover:underline">{metadata.github_repo}</a>
                </div>
            </div>
        </section>
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
    parser = argparse.ArgumentParser(
        description="Generate Project Dashboard index.html"
    )
    parser.add_argument("--output_file", required=True, help="Output HTML file path")

    add_common_args(parser)

    args = parser.parse_args()

    metadata = report_metadata_creation(args)

    html = generate_dashboard_html(metadata)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Project Dashboard generated: {args.output_file}")


if __name__ == "__main__":
    main()
