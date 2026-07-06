SEED_PROMPTS: list[str] = [
    (
        "You are an expert competitive programmer. Before writing any code, "
        "analyse the problem carefully: identify the input format, the output "
        "format, the constraints, and any edge cases (empty input, n=1, maximum "
        "values, overflow risk). Pay close attention to output ordering — the "
        "problem statement specifies whether values should be ascending, descending, "
        "or in some other order; match it exactly. Then write a complete, compilable "
        "C++ program that reads from stdin using cin and writes to stdout using cout. "
        "Include main(). Output only the C++ source code — no explanations, no "
        "markdown commentary."
    ),
    (
        "Full C++ solution. Reads stdin, writes stdout. Include main(). "
        "Compiles with g++ -std=c++23. No commentary, just working code."
    ),
    (
        "Think like a competitive programmer. Identify the algorithm pattern — "
        "greedy, dynamic programming, graph traversal, binary search, math. "
        "Write the most direct correct solution as a complete C++ program. "
        "Use cin/cout, include main(), handle all cases within the constraints. "
        "Output only the program."
    ),
    (
        "Approach this step by step: (1) understand the input-output relationship, "
        "(2) identify the core algorithm, (3) handle boundary conditions, "
        "(4) implement as a complete C++ program with main() that reads from cin "
        "and writes to cout. Output only the C++ source."
    ),
    (
        "Before you write anything, trace through the sample inputs manually. "
        "Confirm you understand what the program should output and why. "
        "Then write a complete C++ program — standard input, standard output, "
        "compiles cleanly with g++ -std=c++23. Just the code."
    ),
    (
        "I need a working solution to this Codeforces problem. Give me the full "
        "C++ source — main function and everything. It should compile and handle "
        "all the edge cases in the constraints. Just the code, nothing else."
    ),
    (
        "You are a careful C++ programmer. Explicitly think about boundary "
        "conditions: n=0, n=1, all elements equal, maximum possible values, "
        "integer overflow. Then write a complete, self-contained C++ program "
        "with main() that handles all of them. Reads from stdin, writes to stdout."
    ),
    (
        "Determine the correct algorithm or data structure before writing a "
        "single line of code. Consider time and space complexity against the "
        "given constraints. Then implement it as a complete C++ program — "
        "include main(), use standard I/O. Output only the code."
    ),
    (
        "Write a complete, compilable C++ program satisfying all of these: "
        "(1) reads all input from stdin via cin, "
        "(2) writes all output to stdout via cout, "
        "(3) compiles without warnings using g++ -O2 -std=c++23, "
        "(4) handles every case within the stated constraints correctly, "
        "(5) produces exactly the expected output format with no extra lines. "
        "Output only the C++ source code."
    ),
    (
        "First mentally sketch the algorithm in plain English until you are "
        "confident it is correct. Then translate it directly into C++. "
        "Give me the entire program — main included — using cin and cout. "
        "No explanation, no pseudocode in the output, just the C++ source."
    ),
    (
        "Read the problem constraints carefully. Choose an algorithm whose "
        "time complexity fits within those constraints — if n can be 10^5, "
        "O(n^2) will TLE. Implement your solution as a complete C++ program "
        "with main() and standard I/O. Return only the code."
    ),
    (
        "Assume the judge's test data includes adversarial inputs at the "
        "extreme boundaries of the constraints. Write a robust, complete C++ "
        "program that handles them all correctly. Reads from stdin, prints to "
        "stdout. Include main(). No commentary."
    ),
    (
        "Write clean, readable C++ code. Use meaningful variable names and keep "
        "the logic easy to follow. Correctness matters more than cleverness. "
        "Give me the complete source file — main function included — that reads "
        "from standard input and writes to standard output."
    ),
    (
        "Expert competitive programmer: analyse the constraints, pick the right "
        "algorithm, then mentally verify your solution against the provided "
        "sample inputs before writing code. Output a complete C++ program — "
        "not a function snippet, the full source with main() — using cin/cout."
    ),
    (
        "Solve this Codeforces problem in C++. Give me the entire program, not "
        "just a function. It needs to compile with g++ and pass all test cases. "
        "Reads from stdin, writes to stdout. Output only the code."
    ),
]

assert len(SEED_PROMPTS) >= 5, f"Need at least 5 seed prompts; got {len(SEED_PROMPTS)}"
