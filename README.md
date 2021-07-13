The checkpoints folder is the v1 version of the model.

v1 model specs:

- 54 countries, south korea a lot
- Generally trained on 350 images per country, no transformations.
- Super overfitted to training dataset.

v2 model specs:

- Removed uganda
- Added horizontal flipping, and some random zoom to the training phase.
- Trained on approx 1000 images per country

To run on screen:
`poetry run python run.py --data data --model mode-v2`

Recording screen occupies two thirds and terminal window one third.
OCR calls cost $1.50 per 1000 requests.
