#/bin/bash

git pull
git add $1/*

git commit -m "$2"
git push origin master
expected "Username for 'https://github.com': "
send "fernandoFernandeSantos"
expected "password"
send "informatica@4"
#vai pedir a senha

exit
