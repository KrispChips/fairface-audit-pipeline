import pandas as pd
import dlib
from predict import detect_face, predidct_age_gender_race, ensure_dir

def convert_to_csv(file_path="provided_labels.xlsx", output_file="standard_labels.csv"):
    # load .xlsx file
    df = pd.read_excel(file_path, skiprows=3)

    # rename 'file' column to 'img_path' 
    if 'file' in df.columns:
        df = df.rename(columns={'file': 'img_path'})

    # modify image path prefix for extra credit file
    if "extra_credit_labels" in file_path:
        df['img_path'] = df['img_path'].apply(lambda x: x.replace("val/", "extra_val/"))

    # save img_path for fairface
    df[['img_path']].to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")


def run_fairface_prediction_from_csv(input_csv_path="standard_labels.csv",
                                     output_csv_path="test_outputs.csv",
                                     output_face_dir="detected_faces"):
    
    dlib.DLIB_USE_CUDA = False
    print("🚀 Running FairFace prediction pipeline")
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)

    ensure_dir(output_face_dir)
    imgs = pd.read_csv(input_csv_path)['img_path']

    detect_face(imgs, output_face_dir)
    print(f"✅ Detected faces saved at: {output_face_dir}")

    predidct_age_gender_race(output_csv_path, output_face_dir)
    print(f"✅ Prediction CSV saved at: {output_csv_path}")


def make_audit_file(input_excel_path="provided_labels.xlsx", 
                    prediction_csv_path="test_outputs.csv"):
    
    header_rows = pd.read_excel(input_excel_path, nrows=3, header=None)
    audit_df = pd.read_excel(input_excel_path, skiprows=3)
    if 'file' in audit_df.columns:
        audit_df = audit_df.rename(columns={'file': 'img_path'})
    preds_df = pd.read_csv(prediction_csv_path)

    preds_df['img_path'] = preds_df['face_name_align'].apply(
        lambda x: 'val/' + x.split('\\')[-1].split('_')[0] + '.jpg'
    )

    preds_df = preds_df.rename(columns={
        'race': 'fairface race',
        'race4': 'fairface race4',
        'gender': 'fairface gender',
        'age': 'fairface age'
    })

    merged_df = pd.merge(
        audit_df,
        preds_df[['img_path', 'fairface race', 'fairface race4', 'fairface gender', 'fairface age']],
        on='img_path',
        how='left'
    )

    # reorder columns so "quality" and "notes" at end 
    quality_col = "facial recognition software quality"
    notes_col = "notes on quality rating"
    reordered_cols = [col for col in merged_df.columns if col not in [quality_col, notes_col]]
    reordered_cols += [quality_col, notes_col]

    merged_df = merged_df[reordered_cols]

    return header_rows, merged_df


def score_row(row):
    notes = []
    score = 10

    if pd.isna(row['fairface race']) or pd.isna(row['fairface gender']) or pd.isna(row['fairface age']):
        return 0, "Missing FairFace prediction(s)"
    if row['validated gender'] != row['fairface gender']:
        score -= 2
        notes.append("Gender mismatch")
    else:
        notes.append("Gender match")
    if row['validated race/ethnic group'] != row['fairface race']:
        score -= 4
        notes.append("Race mismatch")
    else:
        notes.append("Race match")
    age_buckets = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    try:
        validated_age = str(row["validated age"]).strip().lower()
        if "more than 70" in validated_age:
            validated_age = "70+"

        true_age = age_buckets.index(validated_age)
        pred_age = age_buckets.index(str(row["fairface age"]).strip())
        age_diff = abs(true_age - pred_age)
    except:
        score -= 4
        notes.append("Age parsing error")
        return max(score, 0), "; ".join(notes)
    if age_diff == 0:
        notes.append("Age match")
    elif age_diff == 1:
        score -= 2
        notes.append("Age off by 1 bucket")
    else:
        score -= 4
        notes.append("Age off by 2+ buckets")
        
    return max(score, 0), "; ".join(notes)


def score_fairface_predictions(header_rows, merged_df,
                               output_excel_path="scored_audit_results.xlsx"):
    # apply scoring
    merged_df["facial recognition software quality"] = 0
    merged_df["notes on quality rating"] = ""

    for i, row in merged_df.iterrows():
        score, comment = score_row(row)
        merged_df.at[i, "facial recognition software quality"] = score
        merged_df.at[i, "notes on quality rating"] = comment

    # save final result
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        header_rows.to_excel(writer, index=False, header=False)
        merged_df.to_excel(writer, index=False, startrow=3)

    print(f"✅ Final scored file saved to {output_excel_path}")


def main(input_excel_path="provided_labels.xlsx", predict_csv_path="test_outputs.csv", final_output="scored_audit_results.xlsx"):
    # convert .xlsx to .csv for prediction
    csv_path = "standard_labels.csv"
    convert_to_csv(file_path=input_excel_path, output_file=csv_path)

    # run prediction with predict_csv_path
    prediction_csv_path = predict_csv_path
    run_fairface_prediction_from_csv(input_csv_path=csv_path, output_csv_path=prediction_csv_path)

    # merge prediction results with audit metadata
    header_rows, merged_df = make_audit_file(input_excel_path=input_excel_path, prediction_csv_path=prediction_csv_path)

    # score predictions and write final results to Excel
    final_output_path = final_output
    score_fairface_predictions(header_rows, merged_df, output_excel_path=final_output_path)


if __name__ == "__main__":
    main(input_excel_path="extra_credit_labels.xlsx", predict_csv_path="test_outputs.csv", final_output="scored_audit_results.xlsx")