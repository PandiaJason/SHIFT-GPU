with open('paper/main.tex', 'r') as f:
    content = f.read()

start_marker = r'\begin{figure*}[t]'
end_marker = r'\label{fig:architecture}' + '\n' + r'\end{figure*}'

assert start_marker in content, 'start_marker not found'
assert end_marker in content, 'end_marker not found'

start_idx = content.index(start_marker)
end_idx = content.index(end_marker) + len(end_marker)

with open('paper/test_abstract_fig1.tex', 'r') as f:
    test_tex = f.read()

# Extract tikzpicture code from test_abstract_fig1.tex
tikz_start = r'\begin{tikzpicture}['
tikz_end = r'\end{tikzpicture}'
tikz_code = test_tex[test_tex.index(tikz_start):test_tex.index(tikz_end) + len(tikz_end)]

caption_text = (
    r'\caption{Logical abstract architecture and constrained decision pipeline of the SHIFT-GPU framework. '
    r'(1) Latent encoding maps workload $x$ and hardware candidate $g$ into latent spaces $w_x, h_g \in \mathbb{R}^d$ '
    r'under zero-leakage blinding; (2) Bipartite interaction couples embeddings via multi-head cross-attention, '
    r'residual layer normalization, and FiLM latent modulation ($z^* = \gamma_k \odot z + \beta_k$); '
    r'(3) Physics recombination enforces the analytical Roofline floor ($T_{\text{roof}}$) and thermal envelope ($\mathrm{TDP}$) '
    r'via multiplicative neural residual heads; and (4) Risk calibration and selection evaluates covariate shift $d_{\text{shift}}(g)$, '
    r'bounds tail latency via conformal calibration $\hat{T}_{\text{upper}}$, and identifies the power-minimal feasible accelerator '
    r'$g^* = \arg\min_{g \in \mathcal{G}_{\text{feas}}} \hat{P}(x, g)$ with active probe feedback.}'
)

new_figure = (
    r'\begin{figure*}[t]' + '\n'
    r'\centering' + '\n'
    r'\resizebox{\textwidth}{!}{%' + '\n'
    + tikz_code + '%\n'
    r'}' + '\n'
    + caption_text + '\n'
    r'\label{fig:architecture}' + '\n'
    r'\end{figure*}'
)

new_content = content[:start_idx] + new_figure + content[end_idx:]

with open('paper/main.tex', 'w') as f:
    f.write(new_content)

print('Successfully updated Figure 1 in paper/main.tex without bash expansion')
