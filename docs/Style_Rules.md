# Coding-Sytle

* Classes: Pascal Case
* Variables: lower case/separated by underscore, leading underscore for private variables
* Methods: lower case/separated by underscore, meaningful names (aka what is the function doing)
* Comments: only when necessary, lower case (KISS) <br>
  * no comments inside a function, unless the code is special and hard to understand<br>
  * short functions: short explanation comment<br>
  * big functions: Docstring (https://www.python.org/dev/peps/pep-0257/)
* separate all operators with spaces (e.g.: $1 + 2 = 3$)
* function calls directly followed by parentheses (e.g. $func(param)$)
* avoid very long code lines (> 80 characters), break line instead at an appropriate position
       
This covers only the most important rules to follow. Please refer to 
https://www.python.org/dev/peps/pep-0008/ for detailed information.

# Pull Requests

* Only if the Code is working
* Issue numbers in title
* Code based on coding-style Rules
* no Pull Request to Master Branch!
* Every task in new Branch

# Commits

* If exists, issue number

# Tests

* Every function needs min. one test.
* Test much cases as possible
* Try to unit-test and prevent integration-tests
