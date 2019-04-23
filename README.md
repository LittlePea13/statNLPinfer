# statNLPinfer
Run first download_data.sh
And install the environment from environment.yml

then

	source activate statnlp

Then to train a model, run

	python main.py --encoder_type LSTMEncoder

Options for encoders are:

	'BiLSTMMaxPoolEncoder',
	'LSTMEncoder',
	'BiLSTMEncoder',
	'MeanEmbedding'

For the infer task, 

	python infer.py --custom True --hypothesis='I went to college' --premise='I have an university degree' --model_path 'results/final_LSTMEncoder_2048D.pt'

Or to run on the test set:

	python infer.py --custom False --model_path 'results/final_LSTMEncoder_2048D.pt'
