"""Marginal effects at observed moderator quantiles with cluster delta-method SEs."""
from common import CONTROLS, load_panel, twfe, write_table
import numpy as np

p = load_panel()
p['k_x_antisys'] = p.k_l1 * p.antisys_l5
r2 = twfe(p, 'rep_phys', ['k_l1', 'antisys_l5', 'k_x_antisys'] + CONTROLS)
h = load_panel((1960, 2024))
h['tax_any_l1'] = h.groupby('iso3').tax_any.shift()
h['k_x_tax'] = h.k_l1 * h.tax_any_l1
r4 = twfe(h[h.multiparty_l1 == 1], 'coup', ['k_l1', 'tax_any_l1', 'k_x_tax'] + ['ln_gdppc', 'urban_share', 'trade_open'])
rows = []
for label, res, focal, moderator, product in [
    ('H2 capital at threat quantiles', r2, 'k_l1', 'antisys_l5', 'k_x_antisys'),
    ('H4 capital at tax quantiles', r4, 'k_l1', 'tax_any_l1', 'k_x_tax'),
    ('H4 tax at capital quantiles', r4, 'tax_any_l1', 'k_l1', 'k_x_tax')]:
    d = res._d
    for q in (.25, .5, .75):
        level = float(d[moderator].quantile(q))
        contrast = np.zeros(len(res.params))
        contrast[list(res.params.index).index(focal)] = 1.
        contrast[list(res.params.index).index(product)] = level*d[focal].std()/d[product].std()
        b = float(contrast @ res.params)
        se = float(np.sqrt(contrast @ res.cov_params() @ contrast))
        rows.append([label, f'{q:.2f}', f'{level:.3f}', f'{b:.3f}', f'{se:.3f}',
                     f'[{b-1.96*se:.3f}, {b+1.96*se:.3f}]', str(len(d))])
write_table(rows, ['quantile', 'moderator level', 'marginal effect', 'SE', 'normal 95% interval', 'N'],
            'tab_marginal_effects', 'Marginal effects implied by capital interactions', 'tab:marginal',
            'Change per estimation-sample SD of the focal regressor, holding the moderator fixed at its observed quantile. '
            'The product and main regressors are standardized separately; the contrast reverses that scaling before differentiation. '
            'Cluster delta-method SEs; normal intervals are approximate with few countries. H2 outcome: physical-repression index points; '
            'H4 outcome: coup probability units (multiply by 100 for percentage points). Not causal effects.')
print('done: marginal effects')
