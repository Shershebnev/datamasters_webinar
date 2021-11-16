import albumentations as A
from classification_models.tfkeras import Classifiers

SEED = 42
IDX2LABEL = {0: 'барсук',
             1: 'белка',
             2: 'бурундук',
             3: 'бурый_медведь',
             4: 'гималайский_медведь',
             5: 'енотовидная_собака',
             6: 'заяц',
             7: 'кабан',
             8: 'кабарга',
             9: 'колонок',
             10: 'косуля',
             11: 'леопард',
             12: 'лесной_кот',
             13: 'лиса',
             14: 'люди',
             15: 'мусор',
             16: 'пятнистый_олень',
             17: 'тигр',
             18: 'харза'}
LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}
NUM_CLASSES = len(IDX2LABEL)

AVAILABLE_MODELS = Classifiers.models_names()

AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3)
])

MODEL_DIR = "models"
