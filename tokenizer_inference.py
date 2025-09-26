from athena.tokenizer import tokenizer

if __name__ == "__main__":
    inputs = [
        "Hello World",
        "HelloWorld",
        "fooBarBaz",
        "end-to-end",
        """Making the Case for Action
This fact sheet(pdf) and slide deck provide essential state-specific information that addresses the economic imperative, the equity imperative, and the expectations imperative of the college- and career-ready agenda. These resources can be used on their own or serve as the foundation for a personalized presentation or fact sheet(word), which can be customized with state-specific details and examples. The PowerPoint, in particular, was developed with various users in mind and offers a wide range of case-making data that can be drawn from to support your own advocacy efforts.
Advancing the Agenda
As states continue their efforts to promote college and career readiness, Achieve regularly surveys the states to identify their progress in adopting critical college- and career-ready policies. Below is a summary of Idaho's progress to date:
See Closing the Expectations Gap for more information
State accountability systems focus the efforts of teachers, students, parents, administrators and policymakers to ensure that students and schools meet the established goals, including the goal of ensuring all students graduate ready for college and careers. Idaho has yet to begin to use any of the key college- and career-ready indicators in their accountability system.
|Annual School-level Public Reporting||Statewide Performance Goals||School-level Incentives||Accountability Formula|
|Earning a college- and career-ready diploma|
|Scoring college-ready on a high school assessment|
|Earning college credit while in high school|
|Requiring remedial courses in college|
For an explanation of the indicators, their uses and Achieve’s minimum criteria for college- and career-ready accountability, see here."""
    ]

    for text in inputs:
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(ids)

        print(f"Original : {text}")
        print(f"Num IDs  : {len(ids)}")
        print(f"IDs      : {ids}")
        print(f"Tokens   : {tokens}")
        print(f"Decoded  : {decoded}")
        print(f"Match    : {decoded == text}")
        print("-" * 60)
