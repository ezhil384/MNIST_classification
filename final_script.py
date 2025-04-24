from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

loaded_model = load_model("finalcnnmodel.h5")
print("Model loaded successfully!")

test_data = pd.read_csv("sample_test.csv").values
test_data = test_data / 255.0
test_data = test_data.reshape(-1, 28, 28, 1)

predictions = loaded_model.predict(test_data)
pred_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({'Id': np.arange(1, len(pred_labels)+1), 'Label': pred_labels})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")