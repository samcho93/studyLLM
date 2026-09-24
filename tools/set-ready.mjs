// Mark weeks as published in assets/js/site/weeks.js. Run: node tools/set-ready.mjs 1 2 16
import { readFileSync, writeFileSync } from 'node:fs';
const file = new URL('../assets/js/site/weeks.js', import.meta.url);
let src = readFileSync(file, 'utf8');
for (const n of process.argv.slice(2).map(Number)) {
  const re = new RegExp(`(\{ no: ${n}, [^\n]*?ready: )false`);
  if (!re.test(src)) console.warn('not changed:', n);
  src = src.replace(re, '$1true');
}
writeFileSync(file, src);
