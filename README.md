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


I want to point out that I found the bug (although after deadline). I didn't handle the output of the LSTM properly, I wasn't using only the last output. Or both extremes for the BiLSTM.
