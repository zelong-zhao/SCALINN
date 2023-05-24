import setuptools

setuptools.setup(
    name="SIAM-Transformer",
    author="Zelong Zhao",
    version="1.0.0",
    author_email="zelongzhao@hotmail.com",
    description="SIAM Database Generation tools and Transformer Regression Package.",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    
    scripts=['ML_dmft/bin/ML_dmft_gendb.py',
            'ML_dmft/bin/ML_dmft_fit_truncated_aim_params.py',
            'ML_dmft/bin/ML_dmft_database_post.py',
            'ML_dmft/bin/ML_dmft_merge.py',
            'ML_dmft/bin/ML_dmft_solve_ED.py',
            'ML_dmft/bin/ML_dmft_solve_db.py',
            'ML_dmft/bin/ML_dmft_database_analysis.py',
            
            'ML_dmft/pytorch_bin/andT_main.py',
            'ML_dmft/pytorch_bin/andT_analysis.py',
            'ML_dmft/pytorch_bin/andT_checkpoints_analysis.py',
            ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)