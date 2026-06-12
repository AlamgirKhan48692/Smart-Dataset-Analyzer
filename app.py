from flask import Flask, render_template, url_for, request, redirect, flash, send_file

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, r2_score

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

import os

app = Flask(__name__)

app.secret_key = "mysecretkey"

df = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global df

    if 'dataset' in request.files:

        file = request.files["dataset"]

        if file.filename != "":

            df = pd.read_csv(file)

            flash("File Uploaded Successfully")

            return redirect(url_for("home"))

        else:

            flash("No File Selected")

            return redirect(url_for("home"))

    else:

        flash("No Dataset File Found")

        return redirect(url_for("home"))
    

@app.route("/information")
def information():

    global df

    if df is None:
        flash("Please upload dataset first")

        return redirect(url_for("home"))

    head = df.head(2)

    shape = pd.DataFrame({
        "Property": ["Rows", "Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })

    describe = df.describe()

    # create a temprary memory buffer
    buffer = io.StringIO()

    df.info(buf=buffer)

    info = buffer.getvalue()

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Columns", "Missing Values"]

    has_missing = df.isnull().sum().sum() > 0

    describe_explanation = pd.DataFrame({
        "Statistic": [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
        ],
        "Meaning": [
        "Total non-empty values",
        "Average value",
        "Spread of data from mean",
        "Smallest value",
        "25% of values are below this",
        "Middle value (Median)",
        "75% of values are below this",
        "Largest value"
        ]
    })


    return render_template(
        "information.html",
        head=head,
        shape=shape,
        describe=describe,
        describe_explanation=describe_explanation,
        info=info,
        missing=missing,
        has_missing=has_missing
    )

@app.route('/fillmissing', methods=['GET'])
def fillmissing():
    global df
    if df is None:
        flash("Please upload dataset first")
        return redirect(url_for("home"))
    
    # create a temprary memory buffer
    buffer = io.StringIO()

    df.info(buf=buffer)

    info = buffer.getvalue()

    missing_values = df.isnull().sum()
    missing_values=missing_values[missing_values > 0]

    missing_values = missing_values.reset_index()
    missing_values.columns = ["Columns", "Missing Values"]

    return render_template("fillmissing.html",
                           missing_values=missing_values)

@app.route('/fillcolumns')
def fillcolumns():

    global df

    if df is None:
        flash("Please upload dataset first")
        return redirect(url_for("home"))

    columns = []

    missing_counts = df.isnull().sum()

    total_missing = int(missing_counts.sum())
    affected_columns = int((missing_counts > 0).sum())

    total_rows = len(df)

    for col in missing_counts.index:

        if missing_counts[col] > 0:

            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            missing_pct = round(
                (missing_counts[col] / total_rows) * 100,
                2
            )

            mode_series = df[col].mode()

            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
            else:
                mode_value = "No Mode"

            if is_numeric:

                mean_raw = df[col].mean()
                median_raw = df[col].median()

                mean_value = (
                    round(mean_raw, 2)
                    if pd.notna(mean_raw)
                    else ""
                )

                median_value = (
                    round(median_raw, 2)
                    if pd.notna(median_raw)
                    else ""
                )

                # Suggestion Logic
                if missing_pct < 30:
                    suggestion = f"Mean = {mean_value}"

                elif missing_pct < 80:
                    suggestion = f"Median = {median_value}"

                else:
                    suggestion = "Drop Column"

            else:

                mean_value = ""
                median_value = ""

                if missing_pct < 80:
                    suggestion = f"Mode = {mode_value}"
                else:
                    suggestion = "Drop Column"

            columns.append({
                "name": col,
                "dtype": "Numerical" if is_numeric else "Categorical",
                "missing": int(missing_counts[col]),
                "missing_pct": missing_pct,
                "mean": mean_value,
                "median": median_value,
                "mode": mode_value,
                "suggestion": suggestion
            })

    return render_template(
        "fillcolumns.html",
        columns=columns,
        total_missing=total_missing,
        affected_columns=affected_columns
    )

@app.route('/apply_missing_values', methods=["POST"])
def apply_missing_values():

    global df

    if df is None:
        flash("Please upload dataset first")
        return redirect(url_for("home"))

    # BEFORE
    before_missing = df.isnull().sum()
    before_missing = before_missing[before_missing > 0]
    before_missing = before_missing.reset_index()
    before_missing.columns = ["Column", "Missing Values"]

    # Apply treatments
    for column, method in request.form.items():

        if method == "mean" and pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())

        elif method == "median" and pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

        elif method == "mode":
            df[column] = df[column].fillna(df[column].mode()[0])

        elif method == "ffill":
            df[column] = df[column].ffill()

        elif method == "bfill":
            df[column] = df[column].bfill()

        elif method == "drop":
            df = df.dropna(subset=[column])
        
        elif method == "drop_column":
            df.drop(columns=[column], inplace=True)
    
    if df.empty:
        flash(
            "Dataset became empty after removing rows. "
            "Please choose another treatment method."
        )
        return redirect(url_for("fillcolumns"))

    total_missing = int(df.isnull().sum().sum())

    if total_missing == 0:
        
        flash("Missing values handled successfully.")

    # AFTER
    after_missing = df.isnull().sum().reset_index()
    after_missing.columns = ["Column", "Missing Values"]

    return render_template(
        "apply_missing_values.html",
        before_missing=before_missing,
        after_missing=after_missing
    )

@app.route('/download')
def download():

    global df 

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name="cleaned_dataset.csv"
    )

@app.route('/showcolumns')
def showcolumns():

    numerical_columns = df.select_dtypes(
        include=['int64','float64']
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns.tolist()

    all_columns = df.columns.tolist()

    return render_template(
        "showcolumns.html",
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        all_columns=all_columns
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    global df

    if df is None:

        flash("Please upload dataset first")

        return redirect(url_for("home"))

    select_column = request.form["target_column"]

    # Statistics only for numeric columns
    if not pd.api.types.is_numeric_dtype(df[select_column]):

        flash("Statistics available only for numerical columns")

        return redirect(url_for("showcolumns"))

    mean_value = round(df[select_column].mean(), 2)
    median_value = round(df[select_column].median(), 2)
    mode_value = round(df[select_column].mode()[0], 2)

    # Graph
    label = ["Mean", "Median", "Mode"]
    value = [mean_value, median_value, mode_value]

    plt.figure(figsize=(6,4))
    plt.bar(label, value)
    plt.title("Mean Median Mode")

    graph_path = "static/graph.png"

    plt.savefig(graph_path)

    plt.close()

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    has_missing = not missing_values.empty

    return render_template(
        "meanmedianmode.html",
        column_name=select_column,
        mean=mean_value,
        median=median_value,
        mode=mode_value,
        graph=graph_path,
        has_missing=has_missing
    )

@app.route("/model_result", methods=["POST"])
def train_model():

    global df

    if df is None:
        flash("Please upload dataset first")
        return redirect(url_for("home"))

    if df.empty:
        flash("Dataset is empty after cleaning.")
        return redirect(url_for("home"))

    target_column = request.form["target_column"]

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # Detect problem type
    if not is_numeric_dtype(y):
        problem_type = "classification"

    elif y.nunique() <= 10 and set(y.dropna().unique()).issubset(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        ):
        problem_type = "classification"

    else:
        problem_type = "regression"

    if len(X) == 0 or len(y) == 0:
        flash("Not enough data available for training.")
        return redirect(url_for("showcolumns"))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_features = X.select_dtypes(
        include=['int64','float64']
    ).columns

    categorical_features = X.select_dtypes(
        include=['object']
    ).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    best_score = -999
    best_model_name = ""
    best_predictions = None

    if problem_type == "regression":


        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }

        for name, model in models.items():

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)

            score = r2_score(y_test, pred)

            if score > best_score:
                best_score = score
                best_model_name = name
                best_predictions = pred

    else:

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier()
        }

        for name, model in models.items():

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)

            score = accuracy_score(y_test, pred)

            if score > best_score:
                best_score = score
                best_model_name = name
                best_predictions = pred

    predictions = []

    for actual, predicted in list(zip(y_test, best_predictions))[:50]:
        if pd.api.types.is_numeric_dtype(y):
            predictions.append({
                "actual": round(float(actual), 2),
                "predicted": round(float(predicted), 2)
            })

        else:

            predictions.append({
                "actual": actual,
                "predicted": predicted
            })

    app.config["PREDICTIONS"] = predictions

    return render_template(
        "model_result.html",
        target_column=target_column,
        problem_type=problem_type,
        best_model=best_model_name,
        feature_count=X.shape[1],
        train_rows=len(X_train),
        test_rows=len(X_test),
        score=round(best_score,4),
        prediction_count=len(best_predictions)
    )


@app.route("/predictions")
def predictions():

    data = app.config.get("PREDICTIONS", [])

    return render_template(
        "predictions.html",
        predictions=data
    )


@app.route("/visualization")
def visualization():

    global df

    if df is None:

        flash("Please upload dataset first")

        return redirect(url_for("home"))

    all_columns = df.columns.tolist()

    return render_template(
        "visualization.html",
        all_columns=all_columns
    )


@app.route("/generate_chart", methods=["POST"])
def generate_chart():

    global df

    if df is None:
        flash("Please upload dataset first.")
        return redirect(url_for("visualization"))

    analysis_type = request.form.get("analysis_type")

    graph_path = os.path.join(
        app.root_path,
        "static",
        "chart.png"
    )

    try:
        plt.close("all")

        # ==========================================
        # SINGLE COLUMN ANALYSIS
        # ==========================================
        if analysis_type == "single":

            column = request.form.get("column")

            if not column:
                raise ValueError("Please select a column.")

            plt.figure(figsize=(10, 6))

            series = df[column]

            if pd.api.types.is_datetime64_any_dtype(series):
                chart_type = "line"

            elif pd.api.types.is_numeric_dtype(series):

                unique_count = series.nunique()

                if unique_count <= 2:
                    chart_type = "pie"
                elif unique_count <= 20:
                    chart_type = "bar"
                else:
                    chart_type = "histogram"

            else:

                unique_count = series.nunique()

                if unique_count <= 5:
                    chart_type = "pie"
                elif unique_count <= 30:
                    chart_type = "bar"
                else:
                    chart_type = "barh"

            # HISTOGRAM
            if chart_type == "histogram":

                data = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).dropna()

                plt.hist(
                    data,
                    bins=20,
                    edgecolor="black"
                )

                plt.title(f"Histogram of {column}")

            # BAR CHART
            elif chart_type == "bar":

                counts = (
                    df[column]
                    .astype(str)
                    .value_counts()
                    .head(15)
                )

                plt.bar(
                    counts.index,
                    counts.values
                )

                plt.xticks(rotation=45)
                plt.title(f"Bar Chart of {column}")

            # HORIZONTAL BAR
            elif chart_type == "barh":

                counts = (
                    df[column]
                    .astype(str)
                    .value_counts()
                    .head(20)
                )

                plt.barh(
                    counts.index,
                    counts.values
                )

                plt.title(
                    f"Horizontal Bar Chart of {column}"
                )

            # PIE CHART
            elif chart_type == "pie":

                counts = (
                    df[column]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )

                plt.pie(
                    counts.values,
                    labels=counts.index,
                    autopct="%1.1f%%"
                )

                plt.title(f"Pie Chart of {column}")

            # LINE CHART
            elif chart_type == "line":

                temp = (
                    df[column]
                    .value_counts()
                    .sort_index()
                )

                plt.plot(
                    temp.index,
                    temp.values,
                    marker="o"
                )

                plt.xticks(rotation=45)
                plt.title(f"Line Chart of {column}")

            plt.tight_layout()
            plt.savefig(graph_path, bbox_inches="tight")
            plt.close()

            return render_template(
                "chart_result.html",
                graph="chart.png",
                chart_type=chart_type,
                analysis_type=analysis_type
            )

        # ==========================================
        # RELATIONSHIP ANALYSIS
        # ==========================================
        elif analysis_type == "relationship":

            column1 = request.form.get("column1")
            column2 = request.form.get("column2")

            if not column1 or not column2:
                raise ValueError("Please select two columns.")

            plt.figure(figsize=(10, 6))

            col1_numeric = pd.api.types.is_numeric_dtype(df[column1])
            col2_numeric = pd.api.types.is_numeric_dtype(df[column2])

            # BOTH NUMERIC → SCATTER
            if col1_numeric and col2_numeric:

                relation_df = pd.DataFrame({
                    "x": pd.to_numeric(
                        df[column1],
                        errors="coerce"
                    ),
                    "y": pd.to_numeric(
                        df[column2],
                        errors="coerce"
                    )
                }).dropna()

                plt.scatter(
                    relation_df["x"],
                    relation_df["y"]
                )

                plt.xlabel(column1)
                plt.ylabel(column2)

                plt.title(
                    f"{column1} vs {column2}"
                )

                chart_type = "scatter"

            # NUMERIC + CATEGORICAL
            elif col1_numeric and not col2_numeric:

                temp = (
                    df.groupby(column2)[column1]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )

                plt.bar(
                    temp.index.astype(str),
                    temp.values
                )

                plt.xticks(rotation=45)

                plt.title(
                    f"Average {column1} by {column2}"
                )

                chart_type = "bar"

            # CATEGORICAL + NUMERIC
            elif not col1_numeric and col2_numeric:

                temp = (
                    df.groupby(column1)[column2]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )

                plt.bar(
                    temp.index.astype(str),
                    temp.values
                )

                plt.xticks(rotation=45)

                plt.title(
                    f"Average {column2} by {column1}"
                )

                chart_type = "bar"

            # BOTH CATEGORICAL
            else:

                temp = pd.crosstab(
                    df[column1],
                    df[column2]
                )

                temp.plot(
                    kind="bar",
                    figsize=(10, 6)
                )

                plt.title(
                    f"{column1} vs {column2}"
                )

                plt.xticks(rotation=45)

                chart_type = "grouped_bar"

            plt.tight_layout()
            plt.savefig(
                graph_path,
                bbox_inches="tight"
            )
            plt.close()

            return render_template(
                "chart_result.html",
                graph="chart.png",
                chart_type=chart_type,
                analysis_type=analysis_type
            )

        # ==========================================
        # DATASET ANALYSIS
        # ==========================================
        elif analysis_type == "dataset":

            # HEATMAP
            numeric_df = df.select_dtypes(
                include="number"
            )

            if numeric_df.shape[1] >= 2:

                plt.figure(figsize=(10, 6))

                sns.heatmap(
                    numeric_df.corr(),
                    annot=True,
                    cmap="coolwarm"
                )

                plt.title("Correlation Heatmap")

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        app.root_path,
                        "static",
                        "heatmap.png"
                    )
                )

                plt.close()

            # MISSING VALUES
            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if not missing.empty:

                plt.figure(figsize=(8, 5))

                plt.bar(
                    missing.index,
                    missing.values
                )

                plt.xticks(rotation=45)

                plt.title(
                    "Missing Values Analysis"
                )

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        app.root_path,
                        "static",
                        "missing.png"
                    )
                )

                plt.close()

            # DATA TYPES
            numerical = len(
                df.select_dtypes(
                    include="number"
                ).columns
            )

            categorical = len(
                df.select_dtypes(
                    exclude="number"
                ).columns
            )

            plt.figure(figsize=(6, 6))

            if numerical > 0 and categorical > 0:

                plt.pie(
                    [numerical, categorical],
                    labels=[
                        "Numerical",
                        "Categorical"
                    ],
                    autopct="%1.1f%%"
                )

            else:

                plt.bar(
                    ["Numerical", "Categorical"],
                    [numerical, categorical]
                )

            plt.title(
                "Column Type Distribution"
            )

            plt.savefig(
                os.path.join(
                    app.root_path,
                    "static",
                    "datatype.png"
                )
            )

            plt.close()

            return render_template(
                "dataset_analysis.html",
                show_missing=not missing.empty,
                show_heatmap=numeric_df.shape[1] >= 2
            )

        # ==========================================
        # INVALID ANALYSIS TYPE
        # ==========================================
        else:
            raise ValueError(
                "Invalid analysis type."
            )

    except Exception as e:

        print("Chart Error:", str(e))

        flash(
            f"Chart generation failed: {str(e)}"
        )

        return redirect(
            url_for("visualization")
        )

@app.route("/clear")
def clear():

    global df

    df = None

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)