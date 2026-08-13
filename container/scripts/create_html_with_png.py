#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64, os, datetime, html
import pandas as pd
import numpy as np

def define_notes(args):
  if args is None:
      notes = ''
  else:

      if args.adjustBmaskForFW:
          bmaskLong  = 'For each timepoint, the provided brain mask (cyan contour) was modified by excluding voxels with a free water content equal to 100%. The modified'
          bmaskCross = 'The provided brain mask (cyan contour) was modified by excluding voxels with a free water content equal to 100%. The modified'
      else:
          bmaskLong  = 'The'
          bmaskCross = 'The'

      if args.skeletonMask == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz":
          whichSmask = 'the default white matter skeleton mask "delta-svd_skeletonmask_v1.nii.gz", designed to exclude regions with frequent partial volume effects.'
      else:
          whichSmask = 'a custom (provided by the user) white matter skeleton mask.'

      if len(args.tp)>1:
        notes = [f'DELTA-SVD was conducted longitudinally over {len(args.tp)} timepoints. In short:',
                'A within-subject template was first created by registering the free water-corrected FA images non-linearly across all timepoints. '
                'The template was then registered non-linearly onto FSL\'s FMRIB58_FA template in MNI space and further projected onto that template\'s white matter skeleton.',
                f'{bmaskLong} brain mask and the MD and FW image of each timepoint were then taken along the path of estimated transformations, to get a skeletonised version for each of them. '
                'These transformations introduce interpolations between fore- and background voxels, which can be tracked and, hence, were removed from the skeletonised brain mask to reduce partial volume effects.',
                f'The skeletonised brain masks were intersected across timepoints and also with {whichSmask} '
                'The resulting intersection was considered the final white matter skeleton defining the voxels over which the MD and FW summary statistics were calculated to create the final metrics (see table below). '
                'For quality checking, this final white matter skeleton was also back projected into the MNI space, into the within-subject template space and into the '
                'native space of each timepoint (see red overlays shown above).']
      else:
        notes = ['DELTA-SVD was conducted cross-sectionally for a single timepoint. In short:',
                'The free water-corrected FA image of the single timepoint was registered non-linearly onto FSL\'s FMRIB58_FA template in MNI space and further projected onto that template\'s white matter skeleton.',
                f'{bmaskCross} brain mask and the MD and FW image were then taken along the path of estimated transformations, to get a skeletonised version for each of them. '
                'These transformations introduce interpolations between fore- and background voxels, which can be tracked and, hence, were removed from the skeletonised brain mask to reduce partial volume effects.',
                f'The skeletonised brain mask was intersected with {whichSmask} '
                'The resulting intersection was considered the final white matter skeleton defining the voxels over which the MD and FW summary statistics were calculated to create the final metrics (see table below). '
                'For quality checking, this final white matter skeleton was also back projected into the MNI space and into the '
                'native space of each timepoint (see red overlays shown above).']
      
      notes = '<br>'.join(notes)
      notes = f'<p class="text">{notes}</p>'

      if not args.adjustBmaskForFW or args.skeletonMask != "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz":
        notes += '<p class="textbold">Please note that the above described behaviour deviates from the default behaviour, due to the options chosen by the user.</p>'
        
  return notes
  

def create_html_with_png(fnHTML, fnamesPNG, captions=None, notes=None, df=None, args=None):
    
  current_date = datetime.datetime.now()
  date_string = current_date.strftime("%Y-%m-%d %H:%M:%S")
  title = f'''
    <header class="report-header">
      <h1 class="report-header__title">DELTA-SVD Quality Control Report</h1>
      <p class="report-header__date">created on {date_string}</p>
    </header>
  '''

  if args is not None and args.id is not None:
    page_title = f'{html.escape(str(args.id))} - DELTA-SVD QC Report'
  else:
    page_title = 'DELTA-SVD QC Report'

  if args is None:
    meta_table = ''
    inputs_table = ''
    function_call = ''

  else:
    n_tp = len(args.tp)

    #--- badge revealing on hover the default it deviates from; tabindex makes
    #    that tooltip reachable by keyboard too (see the .tag--tip CSS)
    def custom_tag(default_html):
        return ('<span class="tag tag--alt tag--tip" tabindex="0" '
                f'data-default="Default: {default_html}">custom</span>')

    #--- A path as <code>, directory dimmed and filename emphasised so it stays
    #    scannable when it wraps; <wbr> lets it break at "/" and "_" only.
    def path_code(p):
        def brk(s):
            return html.escape(s).replace('/', '/<wbr>').replace('_', '_<wbr>')
        d, name = os.path.split(p)
        if not d:
            return f'<code>{brk(name)}</code>'
        return (f'<code><span class="path-dir">{brk(d + "/")}</span>'
                f'<span class="path-name">{brk(name)}</span></code>')

    #--- metadata table, sharing the style of the timepoint table below
    meta_rows = []
    if args.id is not None:
        meta_rows.append(('Patient ID', f'<span class="summary__id">{html.escape(str(args.id))}</span>'))
    #--- the release that produced this report; results from different versions
    #    must not be pooled
    version = getattr(args, 'version', None)
    if version:
        meta_rows.append(('DELTA-SVD version', f'<code>{html.escape(str(version))}</code>'))
    if args.skeletonMask == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz":
        meta_rows.append(('Skeleton mask', f'{path_code(os.path.basename(args.skeletonMask))} <span class="tag">default</span>'))
    else:
        meta_rows.append(('Skeleton mask', f'{path_code(args.skeletonMask)} {custom_tag("delta-svd_skeletonmask_v1.nii.gz (white matter skeleton excluding regions with frequent partial volume effects)")}'))
    shells = getattr(args, 'shells', None)
    if shells:
        meta_rows.append(('b-value shells',
                          f'{", ".join(str(s) for s in shells)} s/mm&sup2; (&plusmn;25) '
                          + custom_tag("range 800&ndash;1200 s/mm&sup2;")))
    elif list(args.bRange) != [800, 1200]:
        meta_rows.append(('b-value range', f'{args.bRange[0]}&ndash;{args.bRange[1]} s/mm&sup2; (&plusmn;5) {custom_tag("800&ndash;1200 s/mm&sup2;")}'))
    #--- the angular sampling the tensor and free-water fits had to work with;
    #    below 20 it qualifies every metric in the report, and a warning printed
    #    to a log that is not kept would not travel with the results
    nDirections = getattr(args, 'nDirections', None)
    if nDirections:
        shown = (str(nDirections[0]) if len(set(nDirections)) == 1
                 else ', '.join(f'{n} ({tp})' for n, tp in zip(nDirections, args.tp)))
        caution = ('' if min(nDirections) >= 20 else
                   ' <span class="tag tag--alt tag--tip" tabindex="0" data-default="Below the '
                   'recommended minimum of 20: the free-water fraction and hence the reported '
                   'metrics are noisy. Do not pool with results from data with more '
                   'directions.">few</span>')
        meta_rows.append(('Diffusion directions', f'{shown}{caution}'))
    if args.RmaskMNI is not None:
        meta_rows.append(('ROI mask (MNI)', path_code(args.RmaskMNI)))
    if args.hemispheres:
        meta_rows.append(('Hemispheres', 'left and right analysed separately'))
    if not args.adjustBmaskForFW:
        meta_rows.append(('Brain masks', f'not adjusted, i.e. voxels with 100% free water were not removed {custom_tag("adjusted, i.e. voxels with 100% free water are removed")}'))
    #--- only shown when it leaves the validated 12: it is the one threading
    #    setting that changes the metrics, so a deviation belongs beside them
    itkThreads = getattr(args, 'itkThreads', None)
    if n_tp > 1 and itkThreads is not None and itkThreads != 12:
        meta_rows.append(('ITK threads',
                          f'{itkThreads} per registration job '
                          + custom_tag('12 (the validated value; results from different '
                                       'values must not be pooled)')))
    meta_body = ''.join(f'<tr><td class="key">{k}</td><td>{v}</td></tr>' for k, v in meta_rows)
    meta_table = ('<table class="meta-table">'
                  '<thead><tr><th>Item</th><th>Value</th></tr></thead>'
                  f'<tbody>{meta_body}</tbody></table>')

    has_emask = any(args.Emask[i] for i in range(n_tp))
    has_rmask = any(args.Rmask[i] for i in range(n_tp))
    head = '<th>Timepoint</th><th>DWI image</th><th>Brain mask</th>'
    if has_emask: head += '<th>Exclusion mask</th>'
    if has_rmask: head += '<th>ROI mask</th>'
    body = ''
    for i, tp in enumerate(args.tp):
        row = (f'<td class="key">{html.escape(str(tp))}</td>'
               f'<td>{path_code(args.dwi[i])}</td>'
               f'<td>{path_code(args.bmask[i])}</td>')
        if has_emask:
            row += f'<td>{path_code(args.Emask[i]) if args.Emask[i] else "&ndash;"}</td>'
        if has_rmask:
            row += f'<td>{path_code(args.Rmask[i]) if args.Rmask[i] else "&ndash;"}</td>'
        body += f'<tr>{row}</tr>'
    inputs_table = (
        f'<table class="inputs"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    )

    function_call = f'<p class="label">DELTA-SVD call</p><pre class="code">{html.escape(args.function_call)}</pre>'

  #--- notes, shown below the pictures
  if notes is None:
     notes = define_notes(args)
  else:
     notes = '<br>'.join(notes)
     notes = f'<p class="text">{notes}</p>'
  if notes:
     notes = '<p class="label">Method summary</p>' + notes

  fnCSV = None
  if df is None:
      fnCSV = os.path.splitext(fnHTML)[0] + '.csv'
  elif isinstance(df, str):
      fnCSV = df
  if fnCSV is not None:
    if os.path.exists(fnCSV):
        df = pd.read_csv(fnCSV)
    else:
        df = None
  if df is None:
      dfHTML = ''
  else:
      # mutates the caller's DataFrame in place (drop/assign below too)
      df['value'] = df['value'].astype('object')
      for i,row in df.iterrows():
          if np.mod(row['value'],1)<0.0000001:
            df.loc[i,'value'] = int(row['value'])
      df.drop('skeleton', axis=1, inplace=True)
      dfHTML = df.to_html(classes="mystyle", index=True, index_names=False, border=0)
      dfHTML = '<p class="label">Extracted metrics</p>' + dfHTML

  if captions is None:
     captions = [''] * len(fnamesPNG)
  else:
     captions = [captions[i] if i<len(captions) else '' for i,_ in enumerate(fnamesPNG)]
  captions = [f'<p class="label">{cap}</p>' if len(cap)>0 else '' for cap in captions]

  images = []
  for iPng, fnPNG in enumerate(fnamesPNG):
    with open(fnPNG, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        images.append(f'{captions[iPng]}<img width=100% src="data:image/png;base64,{base64_image}">')
  images = ''.join(images)

  license='<p class="text">Notice: By using DELTA-SVD, you agree to the license terms (CC BY-NC-ND 4.0) described in the <a href="https://github.com/isdneuroimaging/DELTA-SVD/blob/main/LICENSE">LICENSE file</a> at <a href="https://github.com/isdneuroimaging/DELTA-SVD">https://github.com/isdneuroimaging/DELTA-SVD</a></p>'

  style='''
  <style>
    :root {
        --accent: #1f6feb;
        --ink: #1c2430;
        --muted: #5b6675;
        --line: #e3e7ee;
        --code-bg: #f2f4f7;
    }

    * {
        box-sizing: border-box;
    }

    body {
        font-size: 10pt;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        line-height: 1.35;
        color: var(--ink);
        background-color: #ffffff;
        margin: 0;
        padding: 16px;
    }
    p { margin: 0 0 6px; }

    /* Header (flat, no box, no rule) */
    .report-header {
        margin: 0 0 14px;
    }
    .report-header__title {
        margin: 0;
        font-size: 15pt;
        font-weight: 700;
    }
    .report-header__date {
        margin: 2px 0 0;
        font-size: 9pt;
        color: var(--muted);
    }

    /* Stacked sections (no boxes) */
    section {
        margin: 0 0 16px;
    }

    .label {
        font-weight: 700;
        color: var(--muted);
        margin: 18px 0 4px;
    }
    .text {
        max-width: 65em;
        word-wrap: break-word;
    }
    .textbold {
        max-width: 65em;
        word-wrap: break-word;
        font-weight: 700;
        color: #c0362c;
    }

    /* Patient ID emphasis (inside the metadata table) */
    .summary__id {
        font-weight: 700;
    }

    /* Tags */
    .tag {
        display: inline-block;
        font-size: 8.5pt;
        font-weight: 700;
        padding: 0 8px;
        border-radius: 11px;
        background: #e7f0ff;
        color: var(--accent);
        vertical-align: middle;
    }
    .tag--alt {
        background: #fdecd6;
        color: #b5651d;
    }
    /* Hoverable badge: reveals the default it deviates from */
    .tag--tip {
        position: relative;
        cursor: help;
        border-bottom: 1px dotted currentColor;
    }
    .tag--tip:hover::after,
    .tag--tip:focus::after {
        content: attr(data-default);
        position: absolute;
        left: 0;
        top: calc(100% + 6px);
        width: max-content;
        max-width: 260px;
        white-space: normal;
        text-align: left;
        font-size: 8.5pt;
        font-weight: 400;
        line-height: 1.3;
        color: #ffffff;
        background: var(--ink);
        padding: 5px 8px;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
        z-index: 10;
    }

    /* Code & paths */
    code {
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        font-size: 9pt;
        background: var(--code-bg);
        padding: 1px 5px;
        border-radius: 4px;
    }
    pre.code {
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
        font-size: 9pt;
        background: var(--code-bg);
        padding: 9px 12px;
        border-radius: 6px;
        margin: 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* QC images: full width for inspection */
    .images img {
        display: block;
        width: 100%;
        height: auto;
        margin: 2px 0 12px;
    }
    .images p { margin: 6px 0 3px; }

    .row:after {
        content: "";
        display: table;
        clear: both;
    }
    .column {
        float: left;
        width: 100px;
        padding: 10px;
    }

    /* Tables (metadata + inputs + metrics all share one style) */
    .meta-table, .inputs, .mystyle {
        border-collapse: collapse;
        font-family: inherit;
        margin: 0 0 14px;
    }
    .meta-table th, .meta-table td,
    .inputs th, .inputs td,
    .mystyle th, .mystyle td {
        text-align: left;
        padding: 5px 18px 5px 0;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
    }
    .meta-table thead th,
    .inputs thead th,
    .mystyle thead th {
        font-weight: 700;
        color: var(--muted);
    }
    /* first column emphasis: metadata key, inputs timepoint, pandas row index */
    .meta-table td.key,
    .inputs td.key,
    .mystyle tbody th {
        font-weight: 700;
        white-space: nowrap;
        text-align: left;
    }
    /* Long file paths wrap inside their cell instead of widening the whole
       table past the page (important for printing/PDF, where there is no
       horizontal scroll). The filename is emphasised and the directory dimmed
       so the path stays scannable once it wraps across lines. */
    .inputs td code,
    .meta-table td code {
        overflow-wrap: anywhere;
    }
    .path-dir {
        color: var(--muted);
    }
    .path-name {
        font-weight: 700;
    }

    /* Footer / license notice */
    .footer .text {
        font-size: 8.5pt;
        color: var(--muted);
    }
  </style>
  '''

  html_doc=f'''
  <!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{page_title}</title>
      {style}
    </head>
    <body>
      {title}
      <section class="meta">
        {meta_table}
        {inputs_table}
        {function_call}
      </section>
      <section class="images">{images}</section>
      <section class="notes">{notes}</section>
      <section class="metrics">{dfHTML}</section>
      <section class="footer">{license}</section>
    </body>
  </html>
  '''
  
  with open(fnHTML, 'w') as fp:
      fp.write(html_doc)