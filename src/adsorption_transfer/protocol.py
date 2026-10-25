"""Fixed material/condition-block splits; no outcome is inspected."""
import hashlib,json
import numpy as np
import pandas as pd
REQUIRED=['excel_row','material_group','condition_block','duplicate_group']
MAX_K=5

def _block_hash(seed, material, block):
    token=f'transfer-v2|seed={seed}|material={material}|block={block}'
    return hashlib.sha256(token.encode('utf-8')).hexdigest()

def prepare_protocol(df, seed=3407, repeats=5):
    """Return (row_roles, split_summary, config) without looking at outcomes.

    `row_roles` is long-form: one row per input row per repeat/target material.
    Columns: scene_id, repeat, seed, target_material, excel_row, material_group,
    condition_block, duplicate_group, role. Roles are source/query/candidate.

    A target-material query receives min(max(2,floor(n_blocks/3)),n_blocks-5)
    blocks. At least five distinct candidate blocks and one query block are
    required. Current data have >=7 blocks/material, so query has >=2 blocks.
    The remaining source domain is the other materials, with all their rows.
    """
    if not isinstance(repeats,int) or repeats<1:raise ValueError('repeats must be a positive integer')
    if not isinstance(seed,int):raise ValueError('seed must be an integer')
    missing=[c for c in REQUIRED if c not in df]
    if missing:raise ValueError(f'Missing protocol columns: {missing}')
    # Work only with the four permitted metadata columns.
    meta=df.loc[:,REQUIRED].copy().sort_values('excel_row').reset_index(drop=True)
    if meta.isna().any().any():raise ValueError('Protocol identifiers cannot be missing')
    if meta.excel_row.duplicated().any():raise ValueError('excel_row must uniquely identify source rows')
    meta['material_group']=meta.material_group.astype(str)
    if meta.groupby('duplicate_group').material_group.nunique().max()>1:
        raise ValueError('A duplicate group spans materials; source/target separation would leak')
    if meta.groupby('duplicate_group').condition_block.nunique().max()>1:
        raise ValueError('A duplicate group spans condition blocks; repair grouping before splitting')
    if meta.groupby('condition_block').material_group.nunique().max()>1:
        raise ValueError('condition_block IDs must identify material-specific blocks globally')
    all_roles=[];summaries=[]
    for repeat in range(repeats):
        split_seed=seed+repeat
        for material in sorted(meta.material_group.unique()):
            target=meta[meta.material_group==material]
            blocks=target.condition_block.unique().tolist()
            ng=len(blocks)
            nq=min(max(2,ng//3),ng-MAX_K)
            if ng<MAX_K+1 or nq<1:
                raise ValueError(f'{material}: {ng} blocks cannot supply five candidates plus query')
            # Hash randomization is X/identifier-only, independent of row order.
            ordered=sorted(blocks,key=lambda b:(_block_hash(split_seed,material,b),str(b)))
            query_blocks=set(ordered[:nq])
            candidate_blocks=set(ordered[nq:])
            scene=f'r{repeat:02d}_s{split_seed}__{material}'
            roles=meta.copy()
            is_target=roles.material_group.eq(material)
            roles['role']='source'
            roles.loc[is_target,'role']='candidate'
            roles.loc[is_target & roles.condition_block.isin(query_blocks),'role']='query'
            roles.insert(0,'target_material',material)
            roles.insert(0,'seed',split_seed)
            roles.insert(0,'repeat',repeat)
            roles.insert(0,'scene_id',scene)
            if roles.groupby('duplicate_group').role.nunique().max()!=1:
                raise AssertionError(f'{scene}: duplicate group crossed roles')
            if roles.groupby('condition_block').role.nunique().max()!=1:
                raise AssertionError(f'{scene}: condition block crossed roles')
            counts=roles.groupby('role').size()
            summaries.append({'scene_id':scene,'repeat':repeat,'seed':split_seed,
                              'target_material':material,'n_target_rows':len(target),
                              'n_target_blocks':ng,'n_query_rows':int(counts['query']),
                              'n_query_blocks':len(query_blocks),'n_candidate_rows':int(counts['candidate']),
                              'n_candidate_blocks':len(candidate_blocks),
                              'n_source_rows':int(counts['source']),
                              'n_source_materials':int(roles.loc[roles.role=='source','material_group'].nunique()),
                              'query_block_ids':json.dumps(sorted(query_blocks)),
                              'candidate_block_ids':json.dumps(sorted(candidate_blocks))})
            all_roles.append(roles)
    row_roles=pd.concat(all_roles,ignore_index=True)
    split_summary=pd.DataFrame(summaries)
    metadata_hash=hashlib.sha256(pd.util.hash_pandas_object(meta,index=False).values.tobytes()).hexdigest()
    config={'protocol_version':'fixed-query-v2','status':'predeclared exploratory repeated LOMO',
            'public_interface':'prepare_protocol(df, seed=3407, repeats=5) -> row_roles, split_summary, config',
            'base_seed':seed,'repeats':repeats,'split_seeds':[seed+i for i in range(repeats)],
            'max_k':MAX_K,'supported_budgets':list(range(MAX_K+1)),
            'n_input_rows':len(meta),'n_materials':int(meta.material_group.nunique()),
            'n_scenes':len(split_summary),'n_row_role_records':len(row_roles),
            'metadata_columns_read':REQUIRED,'outcome_columns_read':[],
            'metadata_sha256':metadata_hash,
            'query_block_count':'min(max(2, floor(n_blocks/3)), n_blocks-5)',
            'query_assignment':'first n_query blocks in material-specific seeded SHA256 hash order',
            'stratification':'split separately within each target material; hash-random blocks',
            'distribution_matching_claim':False,
            'candidate_pool':'all other blocks of the target material, fixed before anchors',
            'source_domain':'all rows of the other materials; excludes all target labels',
            'fixed_across':['anchor-selection policies','calibration methods','budgets 0..5'],
            'query_X':'may be known for transductive prediction; never query y for any design/tuning',
            'anchor_policy_default':'select from candidate rows only, at most one row per candidate block',
            'candidate_labels':'only selected anchors may be revealed under the cumulative budget',
            'budget_nesting':'reuse policy prefix at k=0..5 within a scene; do not redraw at each k',
            'source_auxiliary':'source-domain auxiliary labels may all be used, with source-only model selection',
            'source_fit_reuse':'source set is identical across the five repeats of a material; source fits may be cached',
            'metric_scope':'same fixed query rows for every method and budget within each scene',
            'repeated_splits_are_independent_new_materials':False,
            'condition_block_is_original_study_id':False,
            'forbidden':['query-label tuning or early stopping','unselected candidate-label access',
                         'changing query after anchor choice','target labels in source training',
                         'treating 80 scenes as 80 independent materials']}
    return row_roles,split_summary,config

def check_protocol(row_roles, summary):
    checks=[]
    for scene,z in row_roles.groupby('scene_id',sort=False):
        target=z.target_material.iloc[0]
        record={'scene_id':scene,'rows_unique':bool(z.excel_row.is_unique),
                'source_excludes_target':bool((z.loc[z.role=='source','material_group']!=target).all()),
                'target_roles_only_query_candidate':bool(z.loc[z.material_group==target,'role'].isin(['query','candidate']).all()),
                'all_other_materials_source':bool(z.loc[z.material_group!=target,'role'].eq('source').all()),
                'duplicate_group_max_roles':int(z.groupby('duplicate_group').role.nunique().max()),
                'condition_block_max_roles':int(z.groupby('condition_block').role.nunique().max()),
                'n_query_blocks':int(z.loc[z.role=='query','condition_block'].nunique()),
                'n_candidate_blocks':int(z.loc[z.role=='candidate','condition_block'].nunique())}
        record['all_checks_pass']=bool(record['rows_unique'] and record['source_excludes_target'] and
            record['target_roles_only_query_candidate'] and record['all_other_materials_source'] and
            record['duplicate_group_max_roles']==1 and record['condition_block_max_roles']==1 and
            record['n_query_blocks']>=1 and record['n_candidate_blocks']>=MAX_K)
        checks.append(record)
    return pd.DataFrame(checks)
