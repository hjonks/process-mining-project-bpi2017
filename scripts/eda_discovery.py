"""

EDA & PROCESS DISCOVERY
BPI 2017: Dutch Bank Loan Application Process

HOW TO RUN:
  cd E:\Projects\Process_Mining_Project\scripts
  python day1_eda_discovery.py

"""

import os
import warnings
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

# Import central path config 
from paths import PATHS, ROOT

# Colour palette  
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

# XES file location
XES_PATH = PATHS['raw'] / 'BPI Challenge 2017.xes'

# STEP 1 — DATA LOADING

def generate_synthetic_bpi2017(n_cases: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic event log mirroring BPI 2017 structure.
    Used automatically when the real XES file has not been downloaded yet.
    Same columns, same process logic, swap in real data by placing the
    XES file at data/raw/BPI Challenge 2017.xes.
    """
    print("  Generating synthetic BPI 2017-style data...")
    random.seed(seed)
    np.random.seed(seed)

    A_ACTS = ['A_Create Application', 'A_Submitted', 'A_Concept', 'A_Accepted',
              'A_Complete', 'A_Denied', 'A_Cancelled', 'A_Pending']
    W_ACTS = ['W_Completeren aanvraag', 'W_Nabellen offertes',
              'W_Valideren aanvraag', 'W_Nabellen incomplete dossiers',
              'W_Afhandelen leads', 'W_Beoordelen fraude']
    O_ACTS = ['O_Create Offer', 'O_Created', 'O_Sent (mail and online)',
              'O_Sent (online only)', 'O_Returned', 'O_Accepted',
              'O_Refused', 'O_Cancelled']

    LOAN_GOALS  = ['Home improvement', 'Liquidation', 'Existing loan takeover',
                   'Refinancing', 'Travel', 'Education', 'Car', 'Other']
    RESOURCES   = [f'User_{i:03d}' for i in range(1, 40)] + ['UNKNOWN', 'SYSTEM']
    APP_TYPES   = ['New credit', 'Limit raise']

    rows      = []
    base_date = datetime(2016, 1, 1)

    for case_id in range(1, n_cases + 1):
        case_name   = f'Application_{case_id:05d}'
        amount      = round(np.random.lognormal(9.5, 0.8))
        loan_goal   = random.choice(LOAN_GOALS)
        app_type    = random.choice(APP_TYPES)
        path_type   = np.random.choice(['normal', 'complex', 'cancelled'],
                                        p=[0.60, 0.25, 0.15])

        if path_type == 'normal':
            activities = (
                ['A_Create Application', 'A_Submitted', 'A_Concept']
                + random.sample(W_ACTS[:3], k=random.randint(1, 3))
                + ['O_Create Offer', 'O_Created', 'O_Sent (mail and online)']
                + random.sample(W_ACTS[3:], k=random.randint(0, 2))
                + random.choice([['O_Accepted', 'A_Accepted', 'A_Complete'],
                                 ['O_Refused', 'A_Denied']])
            )
        elif path_type == 'complex':
            activities = (
                ['A_Create Application', 'A_Submitted', 'A_Concept', 'A_Pending']
                + random.sample(W_ACTS, k=random.randint(3, 5))
                + ['O_Create Offer', 'O_Created', 'O_Sent (mail and online)', 'O_Returned']
                + random.sample(W_ACTS, k=random.randint(2, 4))
                + ['O_Create Offer', 'O_Created', 'O_Sent (online only)']
                + random.choice([['O_Accepted', 'A_Accepted', 'A_Complete'],
                                 ['O_Refused', 'A_Denied']])
            )
        else:
            activities = (
                ['A_Create Application', 'A_Submitted']
                + random.sample(W_ACTS[:2], k=random.randint(0, 2))
                + ['A_Cancelled']
            )

        t = base_date + timedelta(days=random.randint(0, 365),
                                   hours=random.randint(8, 17))
        for act in activities:
            gap = (np.random.exponential(0.5)  if act.startswith('A_') else
                   np.random.exponential(24)   if act.startswith('W_') else
                   np.random.exponential(8))
            t += timedelta(hours=gap)
            rows.append({
                'case:concept:name':    case_name,
                'concept:name':         act,
                'time:timestamp':       t,
                'org:resource':         random.choice(RESOURCES),
                'case:LoanGoal':        loan_goal,
                'case:RequestedAmount': amount,
                'case:ApplicationType': app_type,
                'lifecycle:transition': random.choice(['COMPLETE', 'START']),
            })

    df = pd.DataFrame(rows)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    print(f"  Generated {len(df):,} events across {n_cases:,} cases\n")
    return df


def load_data() -> tuple[pd.DataFrame, bool]:
    
    if XES_PATH.exists():
        print(f"  Loading real XES file: {XES_PATH}")
        try:
            import pm4py
            log = pm4py.read_xes(str(XES_PATH))
            df  = pm4py.convert_to_dataframe(log)
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
            df['time:timestamp'] = df['time:timestamp'].dt.tz_localize(None)
            print(f"  Loaded {len(df):,} events from real XES file\n")
            return df, True
        except Exception as e:
            print(f"  Could not parse XES ({e}) — falling back to synthetic data\n")
    else:
        print(f"  XES file not found at: {XES_PATH}")
        print(f"  Running on SYNTHETIC DATA.")
        print(f"  Download the real file from:")
        print(f"  https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884")
        print(f"  Place it in:  {PATHS['raw']}\n")

    return generate_synthetic_bpi2017(n_cases=5000), False


# STEP 2 — CLEANING & VALIDATION


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    print("STEP 2: CLEANING & VALIDATION")
    print("-" * 50)

    original_len = len(df)

    # Standardise column names across different XES exports
    rename_map = {}
    for col in df.columns:
        if ('case' in col.lower() and 'name' in col.lower()
                and 'concept' in col.lower()):
            rename_map[col] = 'case:concept:name'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop rows missing critical columns
    critical = ['case:concept:name', 'concept:name', 'time:timestamp']
    existing = [c for c in critical if c in df.columns]
    df = df.dropna(subset=existing)

    # Coerce timestamps
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
    df = df.dropna(subset=['time:timestamp'])

    # Remove exact duplicates
    df = df.drop_duplicates(
        subset=['case:concept:name', 'concept:name', 'time:timestamp']
    )

    # Sort chronologically within each case
    df = df.sort_values(
        ['case:concept:name', 'time:timestamp']
    ).reset_index(drop=True)

    print(f"  Original events : {original_len:,}")
    print(f"  Removed (dirty) : {original_len - len(df):,}")
    print(f"  Clean events    : {len(df):,}")
    print(f"  Unique cases    : {df['case:concept:name'].nunique():,}")
    print(f"  Unique activites: {df['concept:name'].nunique():,}")
    print(f"  Date range      : {df['time:timestamp'].min().date()} "
          f"→ {df['time:timestamp'].max().date()}\n")
    return df


# STEP 3 — CASE LEVEL FEATURE ENGINEERING


def engineer_case_features(df: pd.DataFrame) -> pd.DataFrame:
    
    print("STEP 3: CASE-LEVEL FEATURE ENGINEERING")
    print("-" * 50)

    SLA_DAYS = 14

    # Duration
    times = df.groupby('case:concept:name')['time:timestamp'].agg(
        case_start='min', case_end='max'
    )
    times['duration_hours'] = (
        (times['case_end'] - times['case_start']).dt.total_seconds() / 3600
    )
    times['duration_days'] = times['duration_hours'] / 24

    # Event counts
    counts = df.groupby('case:concept:name').agg(
        n_events=('concept:name', 'count'),
        n_unique_activities=('concept:name', 'nunique'),
    )

    # Resource diversity
    if 'org:resource' in df.columns:
        res = (df.groupby('case:concept:name')['org:resource']
                 .nunique().rename('n_resources'))
    else:
        res = pd.Series(0, index=df['case:concept:name'].unique(),
                        name='n_resources')

    # Outcome from last activity
    last = df.groupby('case:concept:name')['concept:name'].last().rename('last_activity')

    def classify_outcome(act):
        act = str(act)
        if act in ['A_Accepted', 'A_Complete']:
            return 'Approved'
        if act in ['A_Denied']:
            return 'Denied'
        if act in ['O_Cancelled']:
            return 'Cancelled'
        if act in ['W_Validate application', 'W_Call after offers',
                   'W_Call incomplete files', 'W_Complete application',
                   'W_Assess potential fraud', 'W_Shortened completion',
                   'W_Personal Loan collection']:
            return 'Still Processing'
        if act in ['O_Sent (mail and online)', 'O_Sent (online only)', 'O_Returned']:
            return 'Offer Sent'
        return 'Unknown'

    # Business attributes
    biz_cols = ['case:LoanGoal', 'case:RequestedAmount', 'case:ApplicationType']
    biz      = df.groupby('case:concept:name')[
        [c for c in biz_cols if c in df.columns]
    ].first()
    if 'case:LoanGoal' in biz.columns:
     biz['case:LoanGoal'] = biz['case:LoanGoal'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Assemble
    case_df = times.join(counts).join(res).join(last).join(biz)
    case_df['outcome']    = case_df['last_activity'].apply(classify_outcome)
    case_df['sla_breach'] = (case_df['duration_days'] > SLA_DAYS).astype(int)
    case_df['sla_days']   = SLA_DAYS

    # Time features
    case_df['start_month']   = case_df['case_start'].dt.month
    case_df['start_quarter'] = case_df['case_start'].dt.quarter
    case_df['start_dow']     = case_df['case_start'].dt.day_name()

    # Readable month name (Jan, Feb... instead of 1, 2...)
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    case_df['start_month_name'] = case_df['start_month'].map(month_map)

    # Duration bucket for Power BI histogram
    def duration_bucket(d):
        if d <= 7:  return '1 - 0-7 days (Fast)'
        if d <= 14: return '2 - 8-14 days (On Time)'
        if d <= 21: return '3 - 15-21 days (Late)'
        if d <= 30: return '4 - 22-30 days (Very Late)'
        return           '5 - 30+ days (Critical)'
    case_df['duration_bucket'] = case_df['duration_days'].apply(duration_bucket)

    # Clean column names (remove case: prefix for Power BI)
    case_df = case_df.rename(columns={
        'case:LoanGoal':        'loan_goal',
        'case:RequestedAmount': 'requested_amount',
        'case:ApplicationType': 'application_type',
    })

    # Format datetimes so Power BI can parse them
    case_df['case_start'] = case_df['case_start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    case_df['case_end']   = case_df['case_end'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"  Cases built       : {len(case_df):,}")
    print(f"  SLA threshold     : {SLA_DAYS} days")
    print(f"  SLA breaches      : {case_df['sla_breach'].sum():,} "
          f"({case_df['sla_breach'].mean()*100:.1f}%)")
    print(f"  Avg duration      : {case_df['duration_days'].mean():.1f} days")
    print(f"  Median duration   : {case_df['duration_days'].median():.1f} days")
    print(f"\n  Outcome breakdown:")
    for outcome, count in case_df['outcome'].value_counts().items():
        print(f"    {outcome:<15} {count:>6,}  ({count/len(case_df)*100:.1f}%)")
    print()

    return case_df


# STEP 4 — PROCESS DISCOVERY


def discover_process(df: pd.DataFrame,
                     case_df: pd.DataFrame) -> tuple[pd.DataFrame,
                                                      pd.Series, pd.Series]:
    
    print("STEP 4: PROCESS DISCOVERY")
    print("-" * 50)

    case_variants   = (df.groupby('case:concept:name')['concept:name']
                         .apply(lambda x: ' → '.join(x))
                         .rename('variant'))
    variant_counts  = case_variants.value_counts()
    activity_freq   = df['concept:name'].value_counts()

    n_variants = len(variant_counts)
    n_cases    = len(case_variants)
    top10_cov  = variant_counts.head(10).sum() / n_cases * 100
    top3_cov   = variant_counts.head(3).sum()  / n_cases * 100

    print(f"  Unique variants   : {n_variants:,}")
    print(f"  Top 3 coverage    : {top3_cov:.1f}% of all cases")
    print(f"  Top 10 coverage   : {top10_cov:.1f}% of all cases")
    print(f"\n  Top 5 variants:")
    for i, (variant, count) in enumerate(variant_counts.head(5).items(), 1):
        short = variant if len(variant) < 80 else variant[:77] + '...'
        print(f"    {i}. ({count:,} cases, {count/n_cases*100:.1f}%) {short}")

    print(f"\n  Top 8 activities by frequency:")
    for act, cnt in activity_freq.head(8).items():
        print(f"    {str(act):<45} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")
    print()

    # Attach variant rank to case table
    case_df = case_df.join(case_variants)
    case_df['variant_rank']      = case_df['variant'].map(
        {v: i + 1 for i, v in enumerate(variant_counts.index)}
    )
    case_df['is_top3_variant']   = (case_df['variant_rank'] <= 3).astype(int)
    case_df['is_top10_variant']  = (case_df['variant_rank'] <= 10).astype(int)

    return case_df, variant_counts, activity_freq


# STEP 5 — VISUALISATIONS


def build_eda_overview_chart(df: pd.DataFrame,
                              case_df: pd.DataFrame,
                              activity_freq: pd.Series) -> Path:
    """
    8-panel summary chart covering duration, outcomes, SLA, activities.
    Saved to: results/figures/eda_overview.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#FAFAFA')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1 — Duration histogram
    ax1 = fig.add_subplot(gs[0, 0])
    durations = case_df['duration_days'].clip(upper=60)
    ax1.hist(durations, bins=40, color=COLORS['secondary'],
             edgecolor='white', linewidth=0.5)
    ax1.axvline(case_df['duration_days'].median(), color=COLORS['accent'],
                linestyle='--', linewidth=2,
                label=f"Median: {case_df['duration_days'].median():.1f}d")
    ax1.axvline(14, color=COLORS['danger'], linestyle='--', linewidth=2,
                label='SLA: 14d')
    ax1.set_title('Case Duration Distribution', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Duration (days, clipped at 60)')
    ax1.set_ylabel('Cases')
    ax1.legend(fontsize=8)

    # 2 — Outcome donut
    ax2 = fig.add_subplot(gs[0, 1])
    oc  = case_df['outcome'].value_counts()
    ax2.pie(oc.values, labels=oc.index, autopct='%1.1f%%',
            colors=[COLORS['success'], COLORS['danger'],
                    COLORS['accent'], COLORS['mid']],
            startangle=90, pctdistance=0.8,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax2.set_title('Case Outcomes', fontweight='bold', fontsize=11)

    # 3 — SLA breach by month
    ax3 = fig.add_subplot(gs[0, 2])
    monthly = (case_df.groupby('start_month')['sla_breach']
                       .mean()
                       .reset_index()
                       .rename(columns={'sla_breach': 'breach_rate'}))
    bars = ax3.bar(monthly['start_month'], monthly['breach_rate'] * 100,
                   color=COLORS['primary'], alpha=0.8, edgecolor='white')
    peak = monthly['breach_rate'].idxmax()
    bars[peak].set_color(COLORS['danger'])
    ax3.set_title('SLA Breach Rate by Month', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Breach Rate (%)')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

    # 4 — Top 10 activities
    ax4 = fig.add_subplot(gs[1, :2])
    top10 = activity_freq.head(10)
    hbars = ax4.barh(range(len(top10)), top10.values,
                     color=COLORS['secondary'], edgecolor='white')
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels([str(a)[:40] for a in top10.index], fontsize=9)
    ax4.set_title('Top 10 Activities by Event Frequency',
                  fontweight='bold', fontsize=11)
    ax4.set_xlabel('Event Count')
    ax4.invert_yaxis()
    for i, val in enumerate(top10.values):
        ax4.text(val + max(top10.values) * 0.005, i,
                 f'{val:,}', va='center', fontsize=8)

    # 5 — Events per case
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(case_df['n_events'].clip(upper=50), bins=30,
             color=COLORS['accent'], edgecolor='white')
    ax5.axvline(case_df['n_events'].median(), color=COLORS['primary'],
                linestyle='--', linewidth=2,
                label=f"Median: {case_df['n_events'].median():.0f}")
    ax5.set_title('Events per Case', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Events (clipped at 50)')
    ax5.set_ylabel('Cases')
    ax5.legend(fontsize=8)

    # 6 — Duration by outcome boxplot
    ax6 = fig.add_subplot(gs[2, 0])
    order   = case_df['outcome'].value_counts().index.tolist()
    grouped = [case_df[case_df['outcome'] == o]['duration_days']
                       .clip(upper=60).values for o in order]
    bp = ax6.boxplot(grouped, labels=order, patch_artist=True)
    bcolors = [COLORS['success'], COLORS['danger'], COLORS['accent'], COLORS['mid']]
    for patch, c in zip(bp['boxes'], bcolors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax6.axhline(14, color=COLORS['danger'], linestyle='--',
                linewidth=1.5, alpha=0.7, label='SLA 14d')
    ax6.set_title('Duration by Outcome', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Outcome')
    ax6.set_ylabel('Duration (days, clipped at 60)')
    ax6.legend(fontsize=8)

    # 7 — SLA breach by loan amount
    ax7 = fig.add_subplot(gs[2, 1])
    if 'case:RequestedAmount' in case_df.columns:
        try:
            bins = pd.qcut(case_df['case:RequestedAmount'], q=5,
                           labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'],
                           duplicates='drop')
            breach_q = case_df.groupby(bins)['sla_breach'].mean() * 100
            ax7.bar(range(len(breach_q)), breach_q.values,
                    color=COLORS['primary'], edgecolor='white')
            ax7.set_xticks(range(len(breach_q)))
            ax7.set_xticklabels(breach_q.index, fontsize=8)
            ax7.set_title('SLA Breach Rate by Loan Amount Quintile',
                          fontweight='bold', fontsize=11)
            ax7.set_xlabel('Loan Amount Quintile')
            ax7.set_ylabel('Breach Rate (%)')
        except Exception:
            ax7.text(0.5, 0.5, 'Insufficient amount data',
                     ha='center', va='center', transform=ax7.transAxes)
    else:
        df_copy = df.copy()
        df_copy['month'] = df_copy['time:timestamp'].dt.to_period('M')
        vol = df_copy.groupby('month').size()
        ax7.plot(range(len(vol)), vol.values,
                 color=COLORS['secondary'], linewidth=2)
        ax7.fill_between(range(len(vol)), vol.values,
                         alpha=0.3, color=COLORS['secondary'])
        ax7.set_title('Event Volume Over Time', fontweight='bold', fontsize=11)
        ax7.set_xlabel('Period')
        ax7.set_ylabel('Events')

    # 8 — Unique activities per case
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.hist(case_df['n_unique_activities'].clip(upper=20), bins=20,
             color=COLORS['purple'], edgecolor='white', alpha=0.85)
    ax8.axvline(case_df['n_unique_activities'].median(),
                color=COLORS['primary'], linestyle='--', linewidth=2,
                label=f"Median: {case_df['n_unique_activities'].median():.0f}")
    ax8.set_title('Unique Activities per Case', fontweight='bold', fontsize=11)
    ax8.set_xlabel('Unique Activity Count (clipped at 20)')
    ax8.set_ylabel('Cases')
    ax8.legend(fontsize=8)

    fig.suptitle(
        'BPI 2017 — Dutch Bank Loan Applications\n'
        'Exploratory Data Analysis & Process Overview',
        fontsize=14, fontweight='bold', y=1.01, color=COLORS['primary']
    )

    out = PATHS['figures'] / 'eda_overview.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved → results/figures/eda_overview.png")
    return out


def build_variant_chart(variant_counts: pd.Series,
                         case_df: pd.DataFrame) -> Path:
    """
    Dedicated variant analysis chart — Pareto curve + SLA breach
    by variant rank.
    Saved to: results/figures/process_variants.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#FAFAFA')

    # Left — Pareto
    ax = axes[0]
    top20 = variant_counts.head(20)
    cumul = top20.cumsum() / variant_counts.sum() * 100
    ax2   = ax.twinx()
    ax.bar(range(len(top20)), top20.values,
           color=COLORS['primary'], alpha=0.75, edgecolor='white')
    ax2.plot(range(len(top20)), cumul.values,
             color=COLORS['accent'], linewidth=2.5,
             marker='o', markersize=5, label='Cumulative %')
    ax2.axhline(80, color=COLORS['danger'], linestyle='--',
                linewidth=1.2, alpha=0.8, label='80% threshold')
    ax.set_title('Process Variant Pareto (Top 20)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Variant Rank (1 = most common)')
    ax.set_ylabel('Case Count', color=COLORS['primary'])
    ax2.set_ylabel('Cumulative Coverage (%)', color=COLORS['accent'])
    ax2.legend(fontsize=9, loc='center right')

    # Right — SLA breach rate by variant rank bucket
    ax3 = axes[1]
    if 'variant_rank' in case_df.columns:
        def rank_bucket(r):
            if r == 1:   return 'Rank 1\n(Most common)'
            if r <= 3:   return 'Rank 2-3'
            if r <= 10:  return 'Rank 4-10'
            if r <= 50:  return 'Rank 11-50'
            return 'Rank 51+'
        case_df = case_df.copy()
        case_df['rank_bucket'] = case_df['variant_rank'].apply(rank_bucket)
        order = ['Rank 1\n(Most common)', 'Rank 2-3', 'Rank 4-10',
                 'Rank 11-50', 'Rank 51+']
        bucket_breach = (case_df.groupby('rank_bucket')['sla_breach']
                                .mean() * 100)
        bucket_breach = bucket_breach.reindex(
            [o for o in order if o in bucket_breach.index]
        )
        colors_v = [COLORS['success'] if v < 10 else
                    COLORS['accent']  if v < 25 else
                    COLORS['danger']
                    for v in bucket_breach.values]
        bars = ax3.bar(range(len(bucket_breach)), bucket_breach.values,
                       color=colors_v, edgecolor='white')
        ax3.set_xticks(range(len(bucket_breach)))
        ax3.set_xticklabels(bucket_breach.index, fontsize=9)
        for i, val in enumerate(bucket_breach.values):
            ax3.text(i, val + 0.5, f'{val:.1f}%', ha='center',
                     fontsize=9, fontweight='bold')
    ax3.set_title('SLA Breach Rate by Variant Rank\n(Rarer variants = worse compliance?)',
                  fontweight='bold', fontsize=12)
    ax3.set_xlabel('Process Variant Rank Bucket')
    ax3.set_ylabel('SLA Breach Rate (%)')

    fig.suptitle(
        'BPI 2017 — Process Variant Analysis\n'
        'How many unique process paths exist, and which are most risky?',
        fontsize=13, fontweight='bold', y=1.02, color=COLORS['primary']
    )
    plt.tight_layout()

    out = PATHS['figures'] / 'process_variants.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved → results/figures/process_variants.png")
    return out



# STEP 5 — SAVE OUTPUTS


def save_processed_data(df: pd.DataFrame,
                         case_df: pd.DataFrame) -> None:
    """Saves cleaned data to data/processed/ for use in Day 2."""
    event_path = PATHS['processed'] / 'event_log_cleaned.csv'
    case_path  = PATHS['processed'] / 'case_features.csv'

    df.to_csv(event_path, index=False)
    case_df.to_csv(case_path)

    print(f"  Saved → data/processed/event_log_cleaned.csv  ({len(df):,} rows)")
    print(f"  Saved → data/processed/case_features.csv      ({len(case_df):,} rows)")


def save_summary_report(df: pd.DataFrame,
                         case_df: pd.DataFrame,
                         variant_counts: pd.Series,
                         activity_freq: pd.Series,
                         is_real: bool) -> None:
    """Saves key statistics and pre-written CV bullets to results/reports/."""
    sla_pct    = case_df['sla_breach'].mean() * 100
    med_days   = case_df['duration_days'].median()
    avg_days   = case_df['duration_days'].mean()
    n_variants = len(variant_counts)
    n_cases    = len(case_df)
    n_events   = len(df)
    top3_cov   = variant_counts.head(3).sum() / n_cases * 100

    report = f"""

BPI 2017 PROCESS MINING — SUMMARY REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data type : {'REAL BPI 2017 (1.2M events)' if is_real else 'SYNTHETIC — download real data from 4TU'}


DATASET
  Total events          : {n_events:,}
  Total cases           : {n_cases:,}
  Unique activities     : {df['concept:name'].nunique()}
  Date range            : {df['time:timestamp'].min().date()} to {df['time:timestamp'].max().date()}

PROCESS COMPLEXITY
  Unique process variants : {n_variants:,}
  Top 3 variants cover    : {top3_cov:.1f}% of all cases
  Remaining {100-top3_cov:.1f}% of cases follow long-tail variants

CASE DURATION
  Average   : {avg_days:.1f} days
  Median    : {med_days:.1f} days
  Min       : {case_df['duration_days'].min():.1f} days
  Max       : {case_df['duration_days'].max():.1f} days

SLA PERFORMANCE  (threshold: 14 days)
  Breach count : {case_df['sla_breach'].sum():,} cases ({sla_pct:.1f}%)
  Met SLA      : {(1 - case_df['sla_breach'].mean())*100:.1f}% of cases

OUTCOME SPLIT
{case_df['outcome'].value_counts().to_string()}

TOP 5 ACTIVITIES
{activity_freq.head(5).to_string()}


"""
    out = PATHS['reports'] / 'process_mining_summary.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved → results/reports/process_mining_summary.txt")
    print(report)


# MAIN


if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  PROCESS MINING PROJECT | EXPLORATORY DATA ANALYSIS & PROCESS DISCOVERY | BPI 2017")
    print("=" * 60)
    print(f"\n  Project root : {ROOT}")
    print()

    df, is_real   = load_data()
    df            = clean_and_validate(df)
    case_df       = engineer_case_features(df)
    case_df, variant_counts, activity_freq = discover_process(df, case_df)

    print("STEP 5: BUILDING VISUALISATIONS")
    print("-" * 50)
    build_eda_overview_chart(df, case_df, activity_freq)
    build_variant_chart(variant_counts, case_df)
    print()

    print("STEP 6: SAVING PROCESSED DATA")
    print("-" * 50)
    save_processed_data(df, case_df)
    print()

    print("STEP 7: SAVING SUMMARY REPORT")
    print("-" * 50)
    save_summary_report(df, case_df, variant_counts, activity_freq, is_real)

    print(f"""
  Outputs written to:

    data/
      processed/
        event_log_cleaned.csv     
        case_features.csv         

    results/
      figures/
        eda_overview.png     
        process_variants.png 
      reports/
        process_mining_summary.txt          

""")
