# English Syllabifier (Eng_Syl)

This Python package defines a wordSegmenter Class that can syllabify English words!
The class uses a Bidirectional LSTM seq2seq model trained on data from the Celex database to provide a best guess segmentation of any given English word.


# Usage

Initialize a Segmenter with

    from eng_syl import wordSegmenter
	syllabifier = wordSegmenter()

Build the inference model by building the base model, loading weights, and building the final model.

		syllabifier.build_training_model()
		syllabifier.load_model_weights()
		syllabifer.build_inference_model()

## Translate

After the inference model is built, segment words with syllabifier.translate()

	    segmented_string = syllabifier.translate("syllable")
	    print(segmented_string)
		
		"syl-la-ble"
