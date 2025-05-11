from shufflenetv2 import ShuffleNetV2
from typing import List

output_classes_names: List[str] = ["Height", "Width", "Length"]

# Base Model train + eval
print("Train and evaluating base model...")
model = ShuffleNetV2(CBAM_status=False, train_enhanced=False)
model.train_model()
model.trained_model.save("trained_basemodel.keras")

metrics = model.evaluate_model_metrics()
for i in range(output_classes_names):
    print(f"Metrics for {output_classes_names[i]}:")
    print(f"MAE: {metrics[i]['mae']}")
    print(f"MSE: {metrics[i]['mse']}")
    print(f"MAPE: {metrics[i]['mape']}%")
    print(f"R-squared: {metrics[i]['r2']}")

# CBAM Model unenhanced train + eval
print("Train and evaluating CBAM model...")
model = ShuffleNetV2(CBAM_status=True, train_enhanced=False)
model.train_model()
model.trained_model.save("trained_CBAMmodel.keras")

metrics = model.evaluate_model_metrics()
for i in range(output_classes_names):
    print(f"Metrics for {output_classes_names[i]}:")
    print(f"MAE: {metrics[i]['mae']}")
    print(f"MSE: {metrics[i]['mse']}")
    print(f"MAPE: {metrics[i]['mape']}%")
    print(f"R-squared: {metrics[i]['r2']}")

# CBAM Model enhanced train + eval
print("Train and evaluating CBAM + image enhancements model...")
model = ShuffleNetV2(CBAM_status=True, train_enhanced=True)
model.train_model()
model.trained_model.save("trained_CBAMEnhancedmodel.keras")

metrics = model.evaluate_model_metrics()
for i in range(output_classes_names):
    print(f"Metrics for {output_classes_names[i]}:")
    print(f"MAE: {metrics[i]['mae']}")
    print(f"MSE: {metrics[i]['mse']}")
    print(f"MAPE: {metrics[i]['mape']}%")
    print(f"R-squared: {metrics[i]['r2']}")
