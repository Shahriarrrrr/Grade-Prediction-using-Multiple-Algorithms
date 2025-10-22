import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

class GradePredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.numeric_features = []
        self.categorical_features = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_original = None
        self.y_resampled = None
        
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        
        X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID"])
        y = df["GRADE"]
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def balance_dataset(self, X, y):
        self.y_original = y
        sm = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = sm.fit_resample(X, y)
        self.y_resampled = y_res
        
        print(f"Original class distribution: {dict(y.value_counts().sort_index())}")
        print(f"Resampled class distribution: {dict(pd.Series(y_res).value_counts().sort_index())}\n")
        
        return X_res, y_res
    
    def train_model(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy_score(y_test, self.y_pred):.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, self.y_pred))
        
        return self.y_pred
    
    def plot_class_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        original_counts = pd.Series(self.y_original).value_counts().sort_index()
        resampled_counts = pd.Series(self.y_resampled).value_counts().sort_index()
        
        colors1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(original_counts)))
        ax1.bar(original_counts.index, original_counts.values, color=colors1, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Grade', fontsize=12, weight='bold')
        ax1.set_ylabel('Count', fontsize=12, weight='bold')
        ax1.set_title('Original Class Distribution (Imbalanced)', fontsize=14, weight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(original_counts.values):
            ax1.text(original_counts.index[i], v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(resampled_counts)))
        ax2.bar(resampled_counts.index, resampled_counts.values, color=colors2, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Grade', fontsize=12, weight='bold')
        ax2.set_ylabel('Count', fontsize=12, weight='bold')
        ax2.set_title('After SMOTE Resampling (Balanced)', fontsize=14, weight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(resampled_counts.values):
            ax2.text(resampled_counts.index[i], v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                    xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test),
                    cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, weight='bold')
        ax1.set_xlabel('Predicted Grade', fontsize=12, weight='bold')
        ax1.set_ylabel('Actual Grade', fontsize=12, weight='bold')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                    xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test),
                    cbar_kws={'label': 'Percentage'}, linewidths=1, linecolor='gray')
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, weight='bold')
        ax2.set_xlabel('Predicted Grade', fontsize=12, weight='bold')
        ax2.set_ylabel('Actual Grade', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        top_features = feature_importance_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax1.barh(range(len(top_features)), top_features['importance'].values, color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'].values)
        ax1.set_xlabel('Importance Score', fontsize=12, weight='bold')
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=14, weight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        cumsum = np.cumsum(feature_importance_df['importance'].values)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2, markersize=6, color='darkblue')
        ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% threshold')
        ax2.set_xlabel('Number of Features', fontsize=12, weight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=12, weight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self):
        grades = np.unique(self.y_test)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for grade in grades:
            y_test_binary = (self.y_test == grade).astype(int)
            y_pred_binary = (self.y_pred == grade).astype(int)
            
            precision_scores.append(precision_score(y_test_binary, y_pred_binary, zero_division=0))
            recall_scores.append(recall_score(y_test_binary, y_pred_binary, zero_division=0))
            f1_scores.append(f1_score(y_test_binary, y_pred_binary, zero_division=0))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(grades))
        width = 0.25
        
        axes[0, 0].bar(x - width, precision_scores, width, label='Precision', color='skyblue', edgecolor='black')
        axes[0, 0].bar(x, recall_scores, width, label='Recall', color='lightcoral', edgecolor='black')
        axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen', edgecolor='black')
        axes[0, 0].set_xlabel('Grade', fontsize=12, weight='bold')
        axes[0, 0].set_ylabel('Score', fontsize=12, weight='bold')
        axes[0, 0].set_title('Precision, Recall, and F1-Score by Grade', fontsize=14, weight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(grades)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        metrics_df = pd.DataFrame({
            'Grade': grades,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores
        })
        
        for idx, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
            color_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(grades)))
            axes[0, 1].barh(np.arange(len(grades)) + idx * 0.25, metrics_df[metric], 
                           height=0.25, label=metric, alpha=0.8)
        
        axes[0, 1].set_yticks(np.arange(len(grades)) + 0.25)
        axes[0, 1].set_yticklabels(grades)
        axes[0, 1].set_xlabel('Score', fontsize=12, weight='bold')
        axes[0, 1].set_ylabel('Grade', fontsize=12, weight='bold')
        axes[0, 1].set_title('Performance Metrics Comparison', fontsize=14, weight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='x', alpha=0.3)
        axes[0, 1].set_xlim([0, 1.1])
        
        support = [np.sum(self.y_test == grade) for grade in grades]
        colors = plt.cm.Set3(np.linspace(0, 1, len(grades)))
        wedges, texts, autotexts = axes[1, 0].pie(support, labels=[f'Grade {g}' for g in grades], 
                                                    autopct='%1.1f%%', colors=colors, startangle=90,
                                                    textprops={'weight': 'bold', 'size': 10})
        axes[1, 0].set_title('Test Set Grade Distribution', fontsize=14, weight='bold')
        
        correct_by_grade = [np.sum((self.y_test == grade) & (self.y_pred == grade)) for grade in grades]
        total_by_grade = [np.sum(self.y_test == grade) for grade in grades]
        accuracy_by_grade = [c/t if t > 0 else 0 for c, t in zip(correct_by_grade, total_by_grade)]
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(grades)))
        bars = axes[1, 1].bar(grades, accuracy_by_grade, color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_xlabel('Grade', fontsize=12, weight='bold')
        axes[1, 1].set_ylabel('Accuracy', fontsize=12, weight='bold')
        axes[1, 1].set_title('Per-Class Accuracy', fontsize=14, weight='bold')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracy_by_grade):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        y_test_bin = label_binarize(self.y_test, classes=self.model.classes_)
        y_score = self.model.predict_proba(self.X_test)
        
        n_classes = len(self.model.classes_)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Grade {self.model.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=3,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
        plt.title('ROC Curves for Multi-Class Classification', fontsize=14, weight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self):
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5,
            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', linewidth=2, markersize=8, label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='green', linewidth=2, markersize=8, label='Cross-Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')
        
        plt.xlabel('Training Set Size', fontsize=12, weight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, weight='bold')
        plt.title('Learning Curve - Model Performance vs Training Size', fontsize=14, weight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_confidence(self):
        y_proba = self.model.predict_proba(self.X_test)
        max_proba = np.max(y_proba, axis=1)
        
        correct = (self.y_pred == self.y_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].hist(max_proba[correct], bins=20, alpha=0.7, color='green', edgecolor='black', label='Correct')
        axes[0].hist(max_proba[~correct], bins=20, alpha=0.7, color='red', edgecolor='black', label='Incorrect')
        axes[0].set_xlabel('Prediction Confidence', fontsize=12, weight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, weight='bold')
        axes[0].set_title('Prediction Confidence Distribution', fontsize=14, weight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)
        
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracy = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy.append(np.mean(correct[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracy.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        colors = plt.cm.RdYlGn(bin_accuracy)
        
        bars = axes[1].bar(bin_centers, bin_accuracy, width=0.08, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        axes[1].set_xlabel('Prediction Confidence', fontsize=12, weight='bold')
        axes[1].set_ylabel('Actual Accuracy', fontsize=12, weight='bold')
        axes[1].set_title('Calibration Curve - Confidence vs Accuracy', fontsize=14, weight='bold')
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_student(self, student_data):
        new_df = pd.DataFrame([student_data])
        new_df = new_df[self.feature_names]
        
        predicted_grade = self.model.predict(new_df)[0]
        prediction_proba = self.model.predict_proba(new_df)[0]

        print("\n" + "=" * 60)
        print("PREDICTION FOR NEW STUDENT")
        print("=" * 60)
        print(f"Predicted Grade: {predicted_grade}")
        print(f"Confidence: {prediction_proba.max():.2%}")
        print("\nAll Grade Probabilities:")
        for grade, prob in sorted(zip(self.model.classes_, prediction_proba)):
            bar = "█" * int(prob * 50)
            print(f"  Grade {grade}: {prob:6.2%} {bar}")
        
        return predicted_grade


def main():
    predictor = GradePredictionModel("StudentsPerformance.csv")
    
    X, y = predictor.load_and_preprocess_data()
    X_res, y_res = predictor.balance_dataset(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    
    predictor.train_model(X_train, y_train)
    y_pred = predictor.evaluate_model(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60 + "\n")
    
    print("1. Class Distribution Analysis...")
    predictor.plot_class_distribution()
    
    print("2. Confusion Matrix Analysis...")
    predictor.plot_confusion_matrix()
    
    print("3. Feature Importance Analysis...")
    predictor.plot_feature_importance(top_n=20)
    
    print("4. Performance Metrics Analysis...")
    predictor.plot_performance_metrics()
    
    print("5. ROC Curves for All Classes...")
    predictor.plot_roc_curves()
    
    print("6. Learning Curve Analysis...")
    predictor.plot_learning_curve()
    
    print("7. Prediction Confidence Analysis...")
    predictor.plot_prediction_confidence()
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)
    
    sample_from_dataset = X.iloc[0].to_dict()
    predictor.predict_new_student(sample_from_dataset)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()