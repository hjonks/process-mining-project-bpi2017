"""
BOTTLENECK ANALYSIS, ROOT CAUSE ML & CONFORMANCE CHECKING
BPI 2017: Dutch Bank Loan Application Process

HOW TO RUN:
  cd E:\Projects\Process_Mining_Project\scripts
  python bottleneck_ml.py
  
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Central path config
from paths import PATHS, ROOT

COLORS = {
    'primary':   '#1A3C5E',
    'secondary': '#2E86AB',
    'accent':    '#F18F01',
    'danger':    '#C73E1D',
    'success':   '#2D6A4F',
    'light':     '#F5F5F5',
    'mid':       '#A8DADC',
    'purple':    '#5C4B8A',
}


# LOAD DATA

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("=" * 60)
    print("  PROCESS MINING PROJECT — BOTTLENECK ANALYSIS & ROOT CAUSE ML")
    print("  Bottleneck · ML · Conformance  |  BPI 2017")
    print("=" * 60)
    print(f"\n  Project root : {ROOT}\n")

    event_path = PATHS['processed'] / 'event_log_cleaned.csv'
    case_path  = PATHS['processed'] / 'case_features.csv'

    if not event_path.exists():
        raise FileNotFoundError(
            f"Event log not found at:\n  {event_path}\n"
            "Run day1_eda_discovery.py first."
        )

    print(f"  Loading event log ...")
    df = pd.read_csv(event_path, parse_dates=['time:timestamp'])

    print(f"  Loading case features ...")
    case_df = pd.read_csv(case_path, index_col=0,
                          parse_dates=['case_start', 'case_end'])

    print(f"  Events loaded : {len(df):,}")
    print(f"  Cases loaded  : {len(case_df):,}\n")
    return df, case_df


# STEP 1 — BOTTLENECK ANALYSIS

def bottleneck_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame,
                                                    pd.DataFrame,
                                                    pd.DataFrame]:
    """
    Calculates inter-activity waiting time for every transition in every case.

    DEFINITIONS:
      waiting_hours  = time elapsed from the END of the previous activity
                       to the START of this activity (queue time)
      bottleneck     = activity with the highest mean_wait × frequency product

    """
    print("STEP 1: BOTTLENECK ANALYSIS")

    df = df.sort_values(['case:concept:name', 'time:timestamp']).copy()
    df['prev_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(1)
    df['prev_activity']  = df.groupby('case:concept:name')['concept:name'].shift(1)
    df['waiting_hours']  = (
        (df['time:timestamp'] - df['prev_timestamp'])
        .dt.total_seconds() / 3600
    )

    transitions = df[
        df['prev_activity'].notna() &
        (df['waiting_hours'] >= 0) &
        (df['waiting_hours'] < 24 * 60)
    ].copy()

    bottleneck = (transitions
        .groupby('concept:name')['waiting_hours']
        .agg(
            mean_wait_hours='mean',
            median_wait_hours='median',
            total_wait_hours='sum',
            n_occurrences='count',
            p90_wait_hours=lambda x: np.percentile(x, 90),
        )
        .reset_index()
    )
    bottleneck['mean_wait_days']   = bottleneck['mean_wait_hours']  / 24
    bottleneck['total_wait_days']  = bottleneck['total_wait_hours'] / 24
    bottleneck['bottleneck_score'] = (
        bottleneck['mean_wait_hours'] * bottleneck['n_occurrences']
    )
    bottleneck = bottleneck.sort_values('bottleneck_score', ascending=False)

    transition_waits = (transitions
        .groupby(['prev_activity', 'concept:name'])['waiting_hours']
        .mean()
        .reset_index()
        .rename(columns={'waiting_hours':  'mean_wait_hours',
                         'prev_activity':  'from_activity'})
    )

    print(f"  {'Activity':<45} {'Mean Wait':>10} {'Count':>8} {'Total Days':>11}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*11}")
    for _, r in bottleneck.head(8).iterrows():
        print(f"  {str(r['concept:name']):<45} "
              f"{r['mean_wait_hours']:>8.1f}h  "
              f"{int(r['n_occurrences']):>7,}  "
              f"{r['total_wait_days']:>10.0f}d")

    out = PATHS['tables'] / 'bottleneck_summary.csv'
    bottleneck.to_csv(out, index=False)
    print(f"\n  Saved → results/tables/bottleneck_summary.csv\n")
    return bottleneck, transitions, transition_waits


# STEP 2 — REWORK DETECTION

def rework_analysis(df: pd.DataFrame,
                    case_df: pd.DataFrame) -> tuple[pd.DataFrame,
                                                     pd.DataFrame]:
    """
    Rework = a case visits the same activity more than once.
    This is pure operational waste, every repeat is a case that couldn't
    move forward and had to be re-processed.
    """
    print("STEP 2: REWORK DETECTION")
    print("-" * 50)

    act_counts = (df
        .groupby(['case:concept:name', 'concept:name'])
        .size()
        .reset_index(name='count')
    )
    rework = act_counts[act_counts['count'] > 1]

    rework_cases = rework['case:concept:name'].nunique()
    total_cases  = df['case:concept:name'].nunique()
    rework_pct   = rework_cases / total_cases * 100

    rework_by_act = (rework
        .groupby('concept:name')
        .agg(cases_with_rework=('case:concept:name', 'nunique'),
             avg_repeats=('count', 'mean'))
        .sort_values('cases_with_rework', ascending=False)
        .head(10)
    )

    has_rework_s = (act_counts.groupby('case:concept:name')['count']
        .max()
        .gt(1)
        .astype(int)
        .rename('has_rework')
    )
    rework_events_s = (act_counts.assign(extra=(act_counts['count'] - 1).clip(lower=0))
        .groupby('case:concept:name')['extra']
        .sum()
        .astype(int)
        .rename('rework_events')
   )
    case_df = case_df.join(has_rework_s).join(rework_events_s)
    case_df['has_rework']    = case_df['has_rework'].fillna(0).astype(int)
    case_df['rework_events'] = case_df['rework_events'].fillna(0).astype(int)
    print(f"  Cases with rework   : {rework_cases:,} ({rework_pct:.1f}%)")
    print(f"  Total rework events : {(rework['count'] - 1).sum():,}")
    print(f"\n  Most reworked activities:")
    print(rework_by_act.to_string())
    print()

    return case_df, rework_by_act


# STEP 3 — ROOT CAUSE ML MODEL

def build_ml_model(case_df: pd.DataFrame) -> tuple:
    """
    Trains 3 classifiers to predict SLA breach and compares them.

    FEATURES (all observable at or near case intake):
      n_events               — how many activities the case has gone through
      n_unique_activities    — how many different activity types
      n_resources            — how many staff members touched it
      has_rework             — any rework loops detected
      rework_events          — number of redundant events
      start_month            — month of submission
      log_requested_amount   — log-transformed loan amount
      outcome_encoded        — approved / denied / cancelled (label encoded)
      dow_encoded            — day of week as integer
      app_type_encoded       — new credit vs limit raise

    TARGET: sla_breach (1 = exceeded 14-day SLA, 0 = met SLA)
    """
    print("STEP 3: ROOT CAUSE ML MODEL")

    feature_cols = []
    df_m = case_df.copy()

    # Numeric
    for col in ['n_events', 'n_unique_activities', 'n_resources',
                'has_rework', 'rework_events', 'start_month']:
        if col in df_m.columns:
            df_m[col] = pd.to_numeric(df_m[col], errors='coerce').fillna(0)
            feature_cols.append(col)

    # Log loan amount
    if 'case:RequestedAmount' in df_m.columns:
        df_m['log_requested_amount'] = np.log1p(
            pd.to_numeric(df_m['case:RequestedAmount'], errors='coerce').fillna(0)
        )
        feature_cols.append('log_requested_amount')

    # Categorical encodes
    if 'outcome' in df_m.columns:
        le = LabelEncoder()
        df_m['outcome_encoded'] = le.fit_transform(df_m['outcome'].fillna('Other'))
        feature_cols.append('outcome_encoded')

    if 'start_dow' in df_m.columns:
        dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                   'Friday':4,'Saturday':5,'Sunday':6}
        df_m['dow_encoded'] = df_m['start_dow'].map(dow_map).fillna(0)
        feature_cols.append('dow_encoded')

    if 'case:ApplicationType' in df_m.columns:
        le2 = LabelEncoder()
        df_m['app_type_encoded'] = le2.fit_transform(
            df_m['case:ApplicationType'].fillna('Unknown')
        )
        feature_cols.append('app_type_encoded')

    X = df_m[feature_cols].fillna(0)
    y = df_m['sla_breach'].fillna(0).astype(int)

    print(f"  Features used : {feature_cols}")
    print(f"  Class balance : Breach {y.sum()} ({y.mean()*100:.1f}%)  "
          f"| No breach {(1-y).sum()} ({(1-y).mean()*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced',
                                       max_iter=500, random_state=42))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results     = {}
    best_auc    = 0
    best_model  = None

    print(f"  {'Model':<25} {'CV AUC':>8} {'Test AUC':>9} {'Test Acc':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*9}")

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)
        test_auc = roc_auc_score(y_test, y_prob)
        test_acc = (y_pred == y_test).mean()

        results[name] = dict(model=model, cv_auc=cv_scores.mean(),
                             cv_std=cv_scores.std(), test_auc=test_auc,
                             test_acc=test_acc, y_pred_proba=y_prob, y_pred=y_pred)
        print(f"  {name:<25} {cv_scores.mean():.4f}   {test_auc:.4f}   {test_acc:.4f}")

        if test_auc > best_auc:
            best_auc   = test_auc
            best_model = name

    print(f"\n  Best model : {best_model}  (AUC = {best_auc:.4f})")

    # Feature importance from Random Forest
    rf = results['Random Forest']['model']
    feat_imp = (pd.DataFrame({'feature': feature_cols,
                               'importance': rf.feature_importances_})
                  .sort_values('importance', ascending=False))

    print(f"\n  Feature importances (Random Forest):")
    for _, row in feat_imp.iterrows():
        bar = 'â–ˆ' * int(row['importance'] * 40)
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    out = PATHS['tables'] / 'feature_importance.csv'
    feat_imp.to_csv(out, index=False)
    print(f"\n  Saved → results/tables/feature_importance.csv\n")

    return results, best_model, feat_imp, feature_cols, X_test, y_test


# STEP 4 — CONFORMANCE CHECKING

def conformance_checking(df: pd.DataFrame,
                          case_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Checks how many cases follow the normative (ideal) process model.

    NORMATIVE RULES:
      1. Case starts with A_Create Application
      2. Case ends with a recognised terminal activity
      3. No rework marker activities present
      4. Application activities broadly precede Offer activities

    RESULT: conformance_rate = % of cases satisfying all 4 rules
    """
    print("STEP 4: CONFORMANCE CHECKING")

    IDEAL_END      = ['A_Complete', 'A_Denied', 'A_Cancelled']
    REWORK_MARKERS = ['A_Pending', 'O_Returned', 'O_Cancelled']

    def check(group):
        acts     = group['concept:name'].tolist()
        acts_set = set(acts)

        starts_ok = len(acts) > 0 and acts[0] == 'A_Create Application'
        ends_ok   = len(acts) > 0 and acts[-1] in IDEAL_END
        no_rework = not any(m in acts_set for m in REWORK_MARKERS)

        a_pos = [i for i, a in enumerate(acts) if a.startswith('A_')]
        o_pos = [i for i, a in enumerate(acts) if a.startswith('O_')]
        order_ok = (not a_pos or not o_pos or min(a_pos) < max(o_pos))

        conformant = starts_ok and ends_ok and no_rework and order_ok
        devs = []
        if not starts_ok: devs.append('bad_start')
        if not ends_ok:   devs.append('bad_end')
        if not no_rework: devs.append('rework_detected')
        if not order_ok:  devs.append('wrong_order')

        return pd.Series({
            'is_conformant':   int(conformant),
            'starts_correctly': int(starts_ok),
            'ends_correctly':   int(ends_ok),
            'no_rework_conf':   int(no_rework),
            'correct_order':    int(order_ok),
            'deviation_types':  ','.join(devs) if devs else 'none',
        })

    print("  Checking conformance for all cases...")
    conformance      = df.groupby('case:concept:name').apply(check)
    conformance_rate = conformance['is_conformant'].mean() * 100

    print(f"\n  Conformance rate      : {conformance_rate:.1f}%")
    print(f"  Non-conformant cases  : {(1-conformance['is_conformant']).sum():,}")

    all_devs   = ','.join(
        conformance[conformance['deviation_types'] != 'none']['deviation_types']
    ).split(',')
    dev_counts = pd.Series(all_devs).value_counts()
    print(f"\n  Deviation breakdown:")
    for dev, cnt in dev_counts.items():
        print(f"    {dev:<25} {cnt:>6,} cases")

    if 'sla_breach' in case_df.columns:
        merged = case_df.join(conformance[['is_conformant']], how='left')
        merged['is_conformant'] = merged['is_conformant'].fillna(0)
        b_c  = merged[merged['is_conformant'] == 1]['sla_breach'].mean()
        b_nc = merged[merged['is_conformant'] == 0]['sla_breach'].mean()
        print(f"\n  SLA breach — conformant cases     : {b_c*100:.1f}%")
        print(f"  SLA breach — non-conformant cases : {b_nc*100:.1f}%")
        if b_c > 0:
            print(f"  Non-conformant are {b_nc/b_c:.1f}x more likely to breach SLA")
    print()
    return conformance, conformance_rate


# STEP 5 — VISUALISATIONS

def build_bottleneck_charts(bottleneck: pd.DataFrame,
                             transitions: pd.DataFrame,
                             rework_by_act: pd.DataFrame,
                             conformance: pd.DataFrame,
                             conformance_rate: float,
                             case_df: pd.DataFrame) -> Path:
    """
    6-panel chart: bottleneck rankings, impact matrix, rework,
    conformance, waiting time distributions.
    Saved to: results/figures/bottleneck_analysis.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#FAFAFA')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1 — Mean wait hours (horizontal bar)
    ax1 = fig.add_subplot(gs[0, :2])
    top10 = bottleneck.head(10).sort_values('mean_wait_hours')
    bar_colors = [COLORS['danger'] if i >= len(top10) - 3
                  else COLORS['secondary'] for i in range(len(top10))]
    bars = ax1.barh(range(len(top10)), top10['mean_wait_hours'],
                    color=bar_colors, edgecolor='white', height=0.7)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels([str(a)[:42] for a in top10['concept:name']], fontsize=9)
    ax1.set_title('Top 10 Bottleneck Activities — Mean Waiting Time (hours)',
                  fontweight='bold', fontsize=11)
    ax1.set_xlabel('Mean Waiting Time (hours)')
    for i, (bar, val) in enumerate(zip(bars, top10['mean_wait_hours'])):
        ax1.text(val + max(top10['mean_wait_hours']) * 0.01, i,
                 f'{val:.1f}h', va='center', fontsize=8)

    # 2 — Bottleneck impact score
    ax2 = fig.add_subplot(gs[0, 2])
    top8 = bottleneck.head(8).sort_values('bottleneck_score')
    ax2.barh(range(len(top8)), top8['bottleneck_score'] / 1000,
             color=COLORS['primary'], edgecolor='white')
    ax2.set_yticks(range(len(top8)))
    ax2.set_yticklabels([str(a)[:28] for a in top8['concept:name']], fontsize=8)
    ax2.set_title("Bottleneck Impact Score\n(Frequency \u00d7 Avg Wait)",
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel("Impact Score ('000s)")

    # 3 — Rework by activity
    ax3 = fig.add_subplot(gs[1, 0])
    if len(rework_by_act) > 0:
        rw = rework_by_act.head(8).sort_values('cases_with_rework')
        ax3.barh(range(len(rw)), rw['cases_with_rework'],
                 color=COLORS['accent'], edgecolor='white')
        ax3.set_yticks(range(len(rw)))
        ax3.set_yticklabels([str(a)[:28] for a in rw.index], fontsize=8)
        ax3.set_title('Activities with Most Rework\n(Cases Revisiting Activity)',
                      fontweight='bold', fontsize=11)
        ax3.set_xlabel('Cases with Rework')

    # 4 — Conformance donut
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.pie(
        [conformance_rate, 100 - conformance_rate],
        labels=[f'Conformant\n{conformance_rate:.1f}%',
                f'Non-conformant\n{100-conformance_rate:.1f}%'],
        colors=[COLORS['success'], COLORS['danger']],
        autopct='',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    ax4.set_title('Process Conformance Rate', fontweight='bold', fontsize=11)

    # 5 — Wait time distribution (top 5 bottlenecks)
    ax5 = fig.add_subplot(gs[1, 2])
    top5 = bottleneck.head(5)['concept:name'].tolist()
    if 'waiting_hours' in transitions.columns:
        cap = transitions['waiting_hours'].quantile(0.95)
        data = [transitions[transitions['concept:name'] == a]['waiting_hours']
                            .clip(upper=cap).values for a in top5]
        bp = ax5.boxplot(data, labels=[str(a)[:18] for a in top5],
                         patch_artist=True)
        box_colors = [COLORS['danger'], COLORS['accent'], COLORS['secondary'],
                      COLORS['primary'], COLORS['mid']]
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax5.set_title('Wait Time Distribution\n(Top 5 Bottlenecks)',
                  fontweight='bold', fontsize=11)
    ax5.set_xlabel('Activity')
    ax5.set_ylabel('Waiting Hours (95th pct clipped)')
    ax5.tick_params(axis='x', rotation=25)

    fig.suptitle(
        'BPI 2017 — Bottleneck Analysis & Process Conformance\n'
        'Where is time lost and how far does reality deviate from ideal?',
        fontsize=13, fontweight='bold', y=1.01, color=COLORS['primary']
    )

    out = PATHS['figures'] / 'bottleneck_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved → results/figures/bottleneck_analysis.png")
    return out


def build_ml_charts(results: dict, best_model: str,
                    feat_imp: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> Path:
    """
    4-panel ML chart: feature importance, ROC curves, confusion matrix.
    Saved to: results/figures/ml_performance.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('#FAFAFA')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # 1 — Feature importance
    ax1 = fig.add_subplot(gs[:, 0:2])
    fi  = feat_imp.sort_values('importance')
    bar_c = [COLORS['danger'] if i >= len(fi) - 3 else COLORS['secondary']
             for i in range(len(fi))]
    bars = ax1.barh(range(len(fi)), fi['importance'],
                    color=bar_c, edgecolor='white')
    ax1.set_yticks(range(len(fi)))
    ax1.set_yticklabels(fi['feature'], fontsize=9)
    ax1.set_title('Random Forest Feature Importance\n'
                  '(Top 3 = strongest SLA breach predictors)',
                  fontweight='bold', fontsize=11)
    ax1.set_xlabel('Importance Score')
    for i, val in enumerate(fi['importance']):
        ax1.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

    # 2 — ROC curves
    ax2 = fig.add_subplot(gs[0, 2:4])
    mcolors = [COLORS['danger'], COLORS['secondary'], COLORS['accent']]
    for (name, res), col in zip(results.items(), mcolors):
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        ax2.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})",
                 color=col, linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (0.500)')
    ax2.set_title('ROC Curves — Model Comparison', fontweight='bold', fontsize=11)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(fontsize=8)

    # 3 — Confusion matrix
    ax3 = fig.add_subplot(gs[1, 2:4])
    cm   = confusion_matrix(y_test, results[best_model]['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Breach', 'Breach'])
    disp.plot(ax=ax3, colorbar=False, cmap='Blues')
    ax3.set_title(f'Confusion Matrix — {best_model}',
                  fontweight='bold', fontsize=11)

    fig.suptitle(
        'BPI 2017 — SLA Breach Root Cause Model\n'
        'What predicts a breach and how accurately can we identify at-risk cases?',
        fontsize=13, fontweight='bold', y=1.01, color=COLORS['primary']
    )

    out = PATHS['figures'] / 'ml_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved → results/figures/ml_performance.png")
    return out


# STEP 6 — SAVE ENRICHED TABLE & REPORT

def save_outputs(case_df: pd.DataFrame,
                 results: dict,
                 best_model: str,
                 feat_imp: pd.DataFrame,
                 bottleneck: pd.DataFrame,
                 conformance_rate: float) -> None:
    """
    Saves the enriched case table and the summary report.
    """
    print("STEP 6: SAVING OUTPUTS")
    print("-" * 50)

    # Enriched case table for Power BI
    out_case = PATHS['tables'] / 'case_features_enriched.csv'
    for col in ['case_start', 'case_end']:
        if col in case_df.columns:
            case_df[col] = pd.to_datetime(case_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    # Strip newlines from all text columns before saving
    for col in case_df.select_dtypes(include='object').columns:
        case_df[col] = case_df[col].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.strip()
    for col in case_df.select_dtypes(include='object').columns:
        case_df[col] = case_df[col].astype(str).str.replace(r'[\r\n\t]+', ' ', regex=True).str.strip()
    case_df = case_df.reset_index()
    case_df.to_csv(out_case, index=False)
    print(f"  Saved → results/tables/case_features_enriched.csv")

    # Summary report
    best          = results[best_model]
    sla_pct       = case_df['sla_breach'].mean() * 100
    rework_pct    = case_df['has_rework'].mean() * 100 if 'has_rework' in case_df.columns else 0
    top_bn_act    = str(bottleneck.iloc[0]['concept:name'])
    top_bn_wait   = bottleneck.iloc[0]['mean_wait_hours']
    top_feature   = feat_imp.iloc[0]['feature']
    top_feat_imp  = feat_imp.iloc[0]['importance']

    report = f"""

BPI 2017 PROCESS MINING — BOTTLENECK SUMMARY REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}

BOTTLENECK ANALYSIS
  Top bottleneck activity : {top_bn_act}
  Mean waiting time       : {top_bn_wait:.1f} hours per occurrence
  Total cumulative delay  : {bottleneck.iloc[0]['total_wait_days']:.0f} days

REWORK ANALYSIS
  Cases with rework loops : {rework_pct:.1f}%
  Total rework events     : {int(case_df['rework_events'].sum()) if 'rework_events' in case_df.columns else 'N/A'}

CONFORMANCE CHECKING
  Conformance rate        : {conformance_rate:.1f}%
  Non-conformance rate    : {100-conformance_rate:.1f}%

ML MODEL PERFORMANCE  (Best: {best_model})
  Cross-validated AUC     : {best['cv_auc']:.4f}
  Test AUC                : {best['test_auc']:.4f}
  Test Accuracy           : {best['test_acc']*100:.1f}%
  Top predictive feature  : {top_feature}  (importance: {top_feat_imp:.3f})

"""
    out_report = PATHS['reports'] / 'bottleneck_summary.txt'
    with open(out_report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved → results/reports/bottleneck_summary.txt")
    print(report)


# MAIN

if __name__ == '__main__':

    df, case_df = load_data()

    bottleneck, transitions, transition_waits = bottleneck_analysis(df)
    case_df, rework_by_act = rework_analysis(df, case_df)
    results, best_model, feat_imp, _, X_test, y_test = build_ml_model(case_df)
    conformance, conformance_rate = conformance_checking(df, case_df)

    print("STEP 5: BUILDING VISUALISATIONS")
    build_bottleneck_charts(bottleneck, transitions, rework_by_act,
                            conformance, conformance_rate, case_df)
    build_ml_charts(results, best_model, feat_imp, X_test, y_test)
    print()

    save_outputs(case_df, results, best_model, feat_imp,
                 bottleneck, conformance_rate)

    print(f"""
  Outputs written to:

    results/
      figures/
        day2_bottleneck_analysis.png  
        day2_ml_performance.png      
      tables/
        bottleneck_summary.csv     
        case_features_enriched.csv   
        feature_importance.csv       
      reports/
        bottleneck_summary.txt              

""")
