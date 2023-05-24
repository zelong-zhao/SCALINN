git pull origin

bash ./patch.sh

pip install .

pushd ~/
python -c 'import ML_dmft; print(ML_dmft.__path__)'
popd

./clean.sh
