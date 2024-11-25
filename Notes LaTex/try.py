def generate_geometry_code(margin, marginparsep, marginparwidth, left):
    """
    Generate LaTeX code for the geometry package with given parameters.

    :param margin: General margin (in inches)
    :param marginparsep: Margin between text and margin notes (in inches)
    :param marginparwidth: Width of the margin notes (in inches)
    :param left: Left margin (in inches)
    :return: LaTeX code as a string
    """
    geometry_code = (
        "\\usepackage["
        f"margin={margin}in, "
        f"marginparsep={marginparsep}in, "
        f"marginparwidth={marginparwidth}in, "
        f"left={left}in"
        "]{{geometry}}"
    )
    return geometry_code


def main():
    print("Enter the values for the geometry package parameters (in inches):")

    try:
        # Prompt user for input values
        margin = float(input("General margin (default 2.6): ") or 2.6)
        marginparsep = float(input("Margin parsep (default 0.3): ") or 0.3)
        marginparwidth = float(input("Margin par width (default 2): ") or 2)
        left = float(input("Left margin (default 1): ") or 1)

        # Generate LaTeX code
        latex_code = generate_geometry_code(margin, marginparsep, marginparwidth, left)
        print("\nGenerated LaTeX code:")
        print(latex_code)

    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()
