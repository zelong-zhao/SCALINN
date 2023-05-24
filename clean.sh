find ./ -name '*.pyc' -delete

find ./ML_dmft -name '*.png' -delete

find . -type d -name __pycache__ -exec rm -r {} \+

rm -rf ./ML_dmft.egg-info
rm -rf ./build
rm -rf ./dist
