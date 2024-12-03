import os

def integrate_code_into_latex(file_path, highlight_lines=None, output_file="output.txt"):
    """
    Reads Python code from a file and integrates it into a LaTeX minted environment.
    
    Args:
        file_path (str): Path to the Python file.
        highlight_lines (list, optional): List of line numbers to highlight in LaTeX. Defaults to None.
        output_file (str): Output file to save the LaTeX code.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the content of the Python file
    with open(file_path, "r") as f:
        code_content = f.read()
    
    # Convert highlight_lines list into a comma-separated string
    highlight_str = ",".join(map(str, highlight_lines)) if highlight_lines else ""
    
    # Generate LaTeX code
    latex_template = r"""
\begin{minted}
        [
        frame=lines,
        framesep=2mm,
        baselinestretch=1.2,
        bgcolor=myblue!20,
        fontsize=\tiny,
        linenos""" + (f",highlightlines={{ {highlight_str} }}" if highlight_str else "") + r"""
        ]
        {python}
""" + code_content + r"""
\end{minted}
"""
    
    # Write LaTeX code to the output file
    with open(output_file, "w") as f:
        f.write(latex_template)
    
    print(f"LaTeX code successfully saved to {output_file}")


file_path = "/home/sean/PhD Project/Coding Projects/The Coding Project/AQCPQCstuff/for_highlighted_code.py"  # Path to your Python file
highlight_lines = []  # Lines to highlight
output_file = "highlighted_code.txt"  # Name of the LaTeX output file

integrate_code_into_latex(file_path, highlight_lines, output_file)
