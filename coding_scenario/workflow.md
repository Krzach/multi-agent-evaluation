Detailed Workflow.
The workflow can be described as follows:

The Commander

- manages and coordinates with two LLM-based assistant agents:
  the Writer and the Safeguard.
- directing the flow of communicatio
- responsibility of handling memory tied to user interactions. This capability enables the Commander to capture and retain valuable context regarding the user’s questions and their corresponding responses. Such memory is subsequently shared across the system, empowering the other agents with context from prior user interactions and ensuring more informed and relevant responses.
  communicate with the Safeguard to screen the code and ascertain its safety
- return the code to the user
- furnishes the user with the concluding answer ( 8 ).

The Writer

- combines the functions of a “Coder” and an “Interpreter” as defined in Li et al. (2023a), will craft code and also interpret execution output logs.

The Safeguard

- screens the code and ascertain its safety

1. receiving the task from user by Commander
2. Passing the question by Commander to Writer
3. The writer generates code and passes it to the commender
4. The Commander communicates with the Safeguard to screen the code and ascertain its safety
5. the code obtains the Safeguard’s clearance to the Commander
6. If at a point there is an exception - either a security red flag raised by Safeguard or code execution failures within Commander, the Commander redirects the issue back to the Writer with essential information in logs
7. If the question requires executing the code and retrieving the results, the Commander uses the returned code, executes it and passes it to the Writer for the results interpretation. The answer is provided by the Writer
8. the Commander furnishes the user with the concluding answer

So, the process from 3 to 6 might be repeated multiple times, until each user query receives a thorough and satisfactory resolution or until the timeout.
