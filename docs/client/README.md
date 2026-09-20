# The documents the client is sent

The getting-started document we email with a build is generated from the two
files here, so it can be corrected when the app changes. The version sent on
2026-09-04 had no source anywhere in this repository: it was written before the
Universal C Runtime was bundled and before `Diagnose.cmd` learned to check the
bootloader files, and it could not be updated to describe either.

| file | what it is |
|---|---|
| `content.json` | the guide's text — the only file to edit |
| `build.js` | renders `content.json` into `Pose3D-Getting-Started.docx` |
| `package.json` | the one dependency, `docx` |
| `RELEASE_NOTES_v1.md` | what to send with the resubmission build |

## Building the guide

```bash
cd docs/client
npm install docx          # writes node_modules/ — never committed
node build.js             # -> Pose3D-Getting-Started.docx
```

Export to PDF from Word or LibreOffice (`soffice --headless --convert-to pdf
Pose3D-Getting-Started.docx`) and send the PDF. Neither the `.docx` nor the
`.pdf` is committed: they are build outputs of `content.json`, and a stale copy
in the repository is exactly the problem this folder exists to remove.

## Editing the content

`content.json` is `{ title, subtitle, intro, sections: [{ heading, blocks }] }`,
where each block is one of `{ "p": "…" }`, `{ "sub": "…" }`,
`{ "bullets": [...] }` or `{ "steps": [...] }`. Inside any text, `**bold**`,
`` `code` `` and bare `http(s)` URLs are rendered as such.

Two things the guide must keep saying, both pinned by tests in
`tests/test_diagnostics.py`:

- the install steps unblock the downloaded zip **before** extracting it, and
  extract to `C:\Pose3D` — the same order as `README.md` and the `README.txt`
  inside the zip;
- the ArUco marker size is typed in **centimetres**, which is what the box in
  the import dialog takes.

Keep it to three pages. It is read by someone who wants to open the app, not a
manual.
