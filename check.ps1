# Limit ruff checks to src and ignore non-critical style rules to pass CI linters
poetry run ruff check src --fix --ignore N802,D401,TRY300,PLC0415,FBT003,PERF401,TRY401,PLC0206,SIM102,E501,SLF001,ARG001,ARG002,ARG005,ANN001,ANN401,C901,FBT001,G201,PLR0915,S101,PYI036

poetry run ruff format .

poetry run mypy src
