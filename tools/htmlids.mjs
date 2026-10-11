/* Minimal tag scanner: pulls every element that carries an id out of the
   built page, so the headless harness runs against the real markup (and
   catches an element the script expects but the page does not define). */
const ATTR = /([a-zA-Z_:][-\w:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;

export function parseHTMLIds(html) {
  const out = new Map();
  const tagRe = /<([a-zA-Z][-\w]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g;
  let m;
  while ((m = tagRe.exec(html))) {
    const tag = m[1];
    if (/^(br|meta|link|style|script|title|head|html|body)$/i.test(tag)) {
      if (tag.toLowerCase() === 'script') continue;
    }
    const attrs = {};
    ATTR.lastIndex = 0;
    let a;
    while ((a = ATTR.exec(m[2]))) {
      const val = a[2] !== undefined ? a[2] : a[3] !== undefined ? a[3] : a[4] !== undefined ? a[4] : true;
      attrs[a[1].toLowerCase()] = val;
    }
    if (attrs.id) out.set(attrs.id, { tag, attrs, raw: m[0] });
  }
  return out;
}
