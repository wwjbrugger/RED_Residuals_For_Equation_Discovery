# Python3 program to convert infix to prefix.

# Function to check if
# given character is
# an operator or not.

class InfixToPrefix():
    def __init__(self, possible_operator_1dim,possible_operator_2dim, possible_operands):
        self.possible_operator_1dim = possible_operator_1dim
        self.possible_operator_2dim = possible_operator_2dim
        self.possible_operator = possible_operator_1dim + possible_operator_2dim
        self.possible_operands=possible_operands
    def isOperator(self, c):
        return c in self.possible_operator


    # Function to find priority
    # of given operator.
    def getPriority(self, C):
        #given A op_1 B op_2 C which operator to evaluate first
        if (C == '-' or C == '+'):
            return 1
        elif (C == '*' or C == '/' or C == 'inv'):
            return 2
        elif (C == 'root' or C == 'sqrt' or C == 'cube' or C == 'square' or C == 'exp' or
            C == 'sin' or C == 'cos' or C == 'tan' or C == 'ln' or C == 'log' or C == 'abs'):
            return 3
        elif (C == '^'):
            return 3
        return 0


    # Function that converts infix
    # expression to prefix expression.
    def infixToPrefix(self, infix):
        # stack for operators.
        operators = []

        # stack for operands.
        operands = []

        for i in range(len(infix)):
            symbol = infix[i]
            # If current character is an
            # opening bracket, then
            # push into the operators stack.
            if (infix[i] == '('):
                operators.append(infix[i])

            # If current character is a
            # closing bracket, then pop from
            # both stacks and push result
            # in operands stack until
            # matching opening bracket is
            # not found.
            elif (infix[i] == ')'):
                while (len(operators) != 0 and operators[-1] != '('):
                    # operand 1
                    self.concat_operator_operands(operands, operators)

                # Pop opening bracket
                # from stack.
                operators.pop()

            # If current character is an
            # operand then push it into
            # operands stack.
            elif (not self.isOperator(infix[i])):
                operands.append(infix[i] + "")

            # If current character is an
            # operator, then push it into
            # operators stack after popping
            # high priority operators from
            # operators stack and pushing
            # result in operands stack.
            else:
                while (len(operators) != 0 and self.getPriority(infix[i]) < self.getPriority(operators[-1])):
                    self.concat_operator_operands(operands, operators)
                operators.append(infix[i])

        # Pop operators from operators
        # stack until it is empty and
        # operation in add result of
        # each pop operands stack.
        while (len(operators) != 0):
            self.concat_operator_operands(operands, operators)

        # Final prefix expression is
        # present in operands stack.
        return operands[-1]


    def concat_operator_operands(self, operands, operators):
        op = operators[-1]
        operators.pop()
        if op in self.possible_operator_2dim:
            op1 = operands[-1]
            operands.pop()
            op2 = operands[-1]
            operands.pop()

            tmp = f"{op} {op2} {op1} "
            operands.append(tmp)

        elif op in self.possible_operator_1dim:
            op1 = operands[-1]
            operands.pop()

            tmp = f"{op} {op1} "
            operands.append(tmp)
        return

    # This code is contributed by decode2207.