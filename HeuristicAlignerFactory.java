package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.BaselineWordAligner;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.util.Pair;

import java.util.HashMap;
import java.util.List;

public class HeuristicAlignerFactory implements WordAlignerFactory
{
	public HeuristicWordAligner newAligner(Iterable<SentencePair> trainingData) {
        HashMap<String, Integer> englishCount = new HashMap<>();
        HashMap<String, Integer> frenchCount = new HashMap<>();
        HashMap<Pair<String, String>, Integer> jointCount = new HashMap<>();
        for(SentencePair datum : trainingData) {
            List<String> englishWords = datum.getEnglishWords();
            List<String> frenchWords = datum.getFrenchWords();

            for(String englishWord : englishWords) {
                englishCount.merge(englishWord, 1, Integer::sum);
            }

            for(String frenchWord : frenchWords) {
                frenchCount.merge(frenchWord, 1, Integer::sum);

                for(String englishWord : englishWords) {
                    jointCount.merge(new Pair(frenchWord, englishWord), 1, Integer::sum);
                }
            }

        }
		 return new HeuristicWordAligner(englishCount, frenchCount, jointCount);
	}
}
