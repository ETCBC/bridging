def show(T, F, ws, getBHSinfo, getOSMinfo, indent='', asString=False):
    '''
    We need to show cases.
    Prints information about the word nodes in list `ws`.
    Which information is printed, depends on the two other arguments,
    which are functions, taking a word node
    and delivering information about that word.
    `getBHSInfo` will be used to grab BHS info, `getOSMinfo` will be used to grab OSM info.
    '''
    t = ('{}{} w{}"{}"\n{}\tBHS: {}\n{}\tOSM: {}'.format(
        indent,
        '{} {}:{}'.format(*T.sectionFromNode(ws[0])),
        '/'.join(str(w) for w in ws),
        '/'.join(F.g_word_utf8.v(w) for w in ws),
        indent,
        '/'.join(getBHSinfo(w) for w in ws),
        indent,
        '/'.join(getOSMinfo(w) for w in ws),
    ))
    if asString:
        return t
    print(t)
