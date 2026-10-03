"""Portable reproduction using OECD public-use files or archived aggregate inputs."""
from pathlib import Path
import argparse, json, sys, shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sses2023_alignment'/'src'))
sys.path.insert(0,str(ROOT/'run4_crosswave_replication'/'src'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-2019',type=Path)
    parser.add_argument('--data-2023',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--aggregate-only',action='store_true',help='Recompute cross-wave statistics from archived site/replicate results.')
    args=parser.parse_args()
    out=ROOT/'reproduced_results'
    legacy=out/'sses2023_alignment'
    current=out/'run4_crosswave_replication'/'exp'
    (legacy/'exp').mkdir(parents=True,exist_ok=True);current.mkdir(parents=True,exist_ok=True)
    if args.aggregate_only:
        for relative in ['sses2023_alignment/exp','run4_crosswave_replication/exp']:
            dest=out/relative
            for p in (ROOT/relative).glob('*.csv'):shutil.copy2(p,dest/p.name)
    else:
        if not args.data_2019 or not args.data_2023:
            parser.error('Both OECD source files are required unless --aggregate-only is used.')
        from prepare_2023 import prepare_wave
        from experiment_replication import run_site_analyses,result_tables
        from experiment_cohort import cohort_differences
        from experiment_transition_matrix import equal_site_transition
        cfg=json.loads((ROOT/'analysis_config.json').read_text())
        cfg['data_2019_path']=str(args.data_2019.resolve())
        cfg['data_2023_path']=str(args.data_2023.resolve())
        transition_rows=[]
        for wave in (2019,2023):
            data,audit=prepare_wave(cfg,wave)
            audit.to_csv(current/f'data_quality_{wave}.csv',index=False)
            full=result_tables(run_site_analyses(data,cfg,wave,False,args.workers),cfg,wave,'site_all')
            cohort=result_tables(run_site_analyses(data,cfg,wave,True,args.workers),cfg,wave,'cohort')
            paths=(
                {'depth':'depth_alignment_2019_reanalysis.csv','curve':'chance_adjusted_alignment_2019_reanalysis.csv','capture':'capacity_inflation_2019_reanalysis.csv','replicates':'replicate_statistics_2019_reanalysis.csv'}
                if wave==2019 else
                {'depth':'depth_alignment_2023.csv','curve':'chance_adjusted_alignment.csv','capture':'capacity_inflation.csv','replicates':'replicate_statistics_2023.csv'}
            )
            for key,name in paths.items():full[key].to_csv(current/name if wave==2019 else legacy/'exp'/name,index=False)
            suffix='2019_reanalysis' if wave==2019 else '2023'
            site,meta=cohort_differences(cohort['depth'],cohort['curve'],cohort['capture'],cohort['replicates'],.5 if wave==2019 else .05)
            site.to_csv(legacy/'exp'/f'cohort_{suffix}.csv',index=False)
            meta.to_csv(legacy/'exp'/f'cohort_meta_{suffix}.csv',index=False)
            for key in ('transition','profiles','nonlinear','diagonal'):
                full[key].to_csv(current/f'{key}_{wave}.csv',index=False)
            t=equal_site_transition(full['transition'])
            selected=t[t['achievement_decile']==1]
            row={'wave':wave}
            for k in (10,20,30,50):
                row[f'achievement_top10_to_development_top{k}']=float(selected[selected['development_decile']<=k//10]['cell_probability'].sum())
            row['achievement_top10_to_development_bottom50']=1-row['achievement_top10_to_development_top50']
            transition_rows.append(row)
        pd.DataFrame(transition_rows).to_csv(current/'top10_transition_crosswave.csv',index=False)
    import common
    common.EXP=current;common.LEGACY=legacy
    import crosswave_cif,crosswave_caa,crosswave_depth,crosswave_cohort
    for module in (crosswave_cif,crosswave_caa,crosswave_depth,crosswave_cohort):module.run()
    for name in ('cif_crosswave_summary.csv','caa_crosswave_curve.csv','depth_alignment_crosswave.csv','cohort_depth_crosswave.csv','cohort_cif_crosswave.csv'):
        expected=pd.read_csv(ROOT/'run4_crosswave_replication'/'exp'/name)
        actual=pd.read_csv(current/name)
        numeric=expected.select_dtypes(include='number').columns
        assert np.allclose(expected[numeric],actual[numeric],equal_nan=True,rtol=1e-8,atol=1e-10),name
        print('Verified',name)
    print('Reproduction complete:',out)

if __name__=='__main__':main()
