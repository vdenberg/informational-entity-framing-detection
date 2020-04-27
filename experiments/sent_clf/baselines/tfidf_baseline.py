from lib.handle_data.LoadData import load_basil_features
from lib.handle_data.SplitData import Split
from lib.classifiers.Baselines import Maj, TFIDFBaseline
from lib.evaluate.CrossValidation import CrossValidation


load = load_basil_features()
cv = CrossValidation()

tfidf_baseline = TFIDFBaseline(basil)
basil['low_tfidf'] = tfidf_baseline.rank_sentences()

maj_baseline = Maj()

spl = Split(basil, which='fan')
data = spl.apply_split(features=['low_tfidf', 'tokens'])
cv = CrossValidation()

averages = cv.cross_validate(data, get_y_pred=tfidf_baseline.get_y_pred)
averages = cv.cross_validate(data, get_y_pred=maj_baseline.get_y_pred)



