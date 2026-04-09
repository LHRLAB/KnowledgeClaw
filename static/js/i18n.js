/**
 * KnowledgeClaw i18n
 * Usage:
 *   t('key')              – translate a key
 *   t('key', {n: 3})      – translate with variable substitution
 *   applyLocale('zh')     – switch language and re-render page text
 *   currentLang()         – returns current language code
 */
(function () {
  const TRANSLATIONS = {
    en: {
      // Nav
      'nav.dashboard': 'Dashboard',
      'nav.graph': 'Graph',
      'nav.chat': 'Chat',
      // Dashboard – stats
      'dashboard.title': 'Knowledge Base Dashboard',
      'dashboard.subtitle': 'Monitor your knowledge graph, ingest new documents, and review activity.',
      'dashboard.stat.files': 'Files Ingested',
      'dashboard.stat.chunks': 'Chunks',
      'dashboard.stat.entities': 'Entity Nodes',
      'dashboard.stat.hyperedges': 'Hyperedge Nodes',
      'dashboard.stat.edges': 'Graph Edges',
      'dashboard.stat.vectors': 'Vector Entries',
      // Dashboard – ingest
      'dashboard.ingest.title': 'Ingest Documents',
      'dashboard.ingest.desc': 'Provide a local folder path to ingest into the knowledge base. Supported: .txt, .md, .pdf, .py, .js, .json, .csv, and more.',
      'dashboard.ingest.label': 'Folder or File Path',
      'dashboard.ingest.placeholder': '/path/to/your/documents',
      'dashboard.ingest.btn': 'Start Ingestion',
      'dashboard.ingest.loading': 'Processing…',
      'dashboard.ingest.complete': 'Ingestion complete',
      'dashboard.ingest.files': 'Files processed',
      'dashboard.ingest.created': 'Chunks created',
      'dashboard.ingest.entities': 'Entities',
      'dashboard.ingest.hedges': 'Hyperedges',
      'dashboard.ingest.errors': 'Errors',
      'dashboard.ingest.failed': 'Ingestion failed. Check server logs.',
      // Dashboard – health
      'dashboard.health.title': 'KB Health Check',
      'dashboard.health.desc': 'Evaluate the health of your knowledge base: connectivity, coverage, and top entities.',
      'dashboard.health.btn': 'Run Evaluation',
      'dashboard.health.loading': 'Evaluating…',
      'dashboard.health.score': 'Health Score',
      'dashboard.health.top': 'Top Entities',
      'dashboard.health.isolated': 'Isolated nodes',
      // Dashboard – quick query
      'dashboard.query.title': 'Quick Query',
      'dashboard.query.fullchat': 'Full Chat →',
      'dashboard.query.placeholder': 'Ask something about your knowledge base…',
      'dashboard.query.ask': 'Ask',
      // Chat – sidebar
      'chat.mode.title': 'Mode',
      'chat.mode.qa': 'Q&A',
      'chat.mode.infer': 'Inference',
      'chat.mode.research': 'Research',
      'chat.desc.title': 'Description',
      'chat.options.title': 'Options',
      'chat.topk': 'Top-K:',
      'chat.rounds': 'Rounds:',
      'chat.clear': 'Clear Chat',
      // Chat – mode metadata
      'chat.mode.qa.label': 'Q&A Mode',
      'chat.mode.qa.hint': 'Using GraphRAG retrieval',
      'chat.mode.qa.desc': 'Answer questions using the knowledge graph. Retrieved context is cited by source.',
      'chat.mode.infer.label': 'Inference Mode',
      'chat.mode.infer.hint': 'ReACT multi-round search',
      'chat.mode.infer.desc': 'Multi-round ReACT inference. The AI thinks, searches, and synthesizes insights from your KB.',
      'chat.mode.research.label': 'Research Mode',
      'chat.mode.research.hint': 'Gap analysis',
      'chat.mode.research.desc': 'Analyse KB coverage of a topic and identify knowledge gaps with expansion suggestions.',
      // Chat – placeholders / dynamic
      'chat.placeholder.qa': 'Ask a question…',
      'chat.placeholder.infer': 'Enter a topic for inference…',
      'chat.placeholder.research': 'Enter a topic to research…',
      'chat.empty': 'Ask a question to get started.',
      'chat.retrieved': '{n} chunk(s) retrieved',
      'chat.searched': 'Searched:',
      'chat.coverage': 'Coverage:',
      'chat.gaps': 'Gaps:',
      'chat.suggestions': 'Suggestions:',
      'chat.error': 'Error:',
      // Dashboard – reset
      'dashboard.reset.title': 'Reset Knowledge Base',
      'dashboard.reset.desc': 'Permanently delete all ingested data: chunks, entities, graph, and vector stores. This cannot be undone.',
      'dashboard.reset.btn': 'Clear All Data',
      'dashboard.reset.done': 'Knowledge base cleared.',
      'dashboard.reset.fail': 'Reset failed. Check server logs.',
      // Graph – toolbar
      'graph.search.placeholder': 'Filter nodes…',
      'graph.reset': 'Reset View',
      'graph.refresh': 'Refresh',
      // Graph – legend
      'graph.legend.entity': 'Entity',
      'graph.legend.hyperedge': 'Hyperedge',
      'graph.legend.unknown': 'Unknown',
      // Graph – details panel
      'graph.details.title': 'Node Details',
      'graph.details.type': 'Type',
      'graph.details.label': 'Label',
      'graph.details.source': 'Source',
      'graph.details.degree': 'Degree',
      'graph.details.connections': 'Connections',
      'graph.details.none': 'No connections',
    },
    zh: {
      // Nav
      'nav.dashboard': '仪表板',
      'nav.graph': '知识图谱',
      'nav.chat': '对话',
      // Dashboard – stats
      'dashboard.title': '知识库仪表板',
      'dashboard.subtitle': '监控知识图谱，导入文档，查看活动记录。',
      'dashboard.stat.files': '已导入文件',
      'dashboard.stat.chunks': '文本块',
      'dashboard.stat.entities': '实体节点',
      'dashboard.stat.hyperedges': '超边节点',
      'dashboard.stat.edges': '图谱边',
      'dashboard.stat.vectors': '向量条目',
      // Dashboard – ingest
      'dashboard.ingest.title': '导入文档',
      'dashboard.ingest.desc': '提供本地文件夹路径以导入知识库。支持：.txt、.md、.pdf、.py、.js、.json、.csv 等。',
      'dashboard.ingest.label': '文件夹或文件路径',
      'dashboard.ingest.placeholder': '/path/to/your/documents',
      'dashboard.ingest.btn': '开始导入',
      'dashboard.ingest.loading': '处理中…',
      'dashboard.ingest.complete': '导入完成',
      'dashboard.ingest.files': '已处理文件',
      'dashboard.ingest.created': '已创建文本块',
      'dashboard.ingest.entities': '实体',
      'dashboard.ingest.hedges': '超边',
      'dashboard.ingest.errors': '错误',
      'dashboard.ingest.failed': '导入失败，请查看服务器日志。',
      // Dashboard – health
      'dashboard.health.title': '知识库健康检查',
      'dashboard.health.desc': '评估知识库健康状况：连通性、覆盖度及核心实体。',
      'dashboard.health.btn': '运行评估',
      'dashboard.health.loading': '评估中…',
      'dashboard.health.score': '健康分数',
      'dashboard.health.top': '核心实体',
      'dashboard.health.isolated': '孤立节点',
      // Dashboard – quick query
      'dashboard.query.title': '快速查询',
      'dashboard.query.fullchat': '完整对话 →',
      'dashboard.query.placeholder': '向知识库提问…',
      'dashboard.query.ask': '提问',
      // Chat – sidebar
      'chat.mode.title': '模式',
      'chat.mode.qa': '问答',
      'chat.mode.infer': '推演',
      'chat.mode.research': '研究',
      'chat.desc.title': '说明',
      'chat.options.title': '选项',
      'chat.topk': 'Top-K：',
      'chat.rounds': '轮次：',
      'chat.clear': '清空对话',
      // Chat – mode metadata
      'chat.mode.qa.label': '问答模式',
      'chat.mode.qa.hint': '使用 GraphRAG 检索',
      'chat.mode.qa.desc': '基于知识图谱回答问题，检索到的上下文将注明来源。',
      'chat.mode.infer.label': '推演模式',
      'chat.mode.infer.hint': 'ReACT 多轮搜索',
      'chat.mode.infer.desc': '多轮 ReACT 推演：AI 思考、搜索并综合知识库洞见。',
      'chat.mode.research.label': '研究模式',
      'chat.mode.research.hint': '缺口分析',
      'chat.mode.research.desc': '分析知识库主题覆盖度，识别知识缺口并提供扩展建议。',
      // Chat – placeholders / dynamic
      'chat.placeholder.qa': '请提问…',
      'chat.placeholder.infer': '输入推演主题…',
      'chat.placeholder.research': '输入研究主题…',
      'chat.empty': '开始提问。',
      'chat.retrieved': '已检索 {n} 个文本块',
      'chat.searched': '已搜索：',
      'chat.coverage': '覆盖度：',
      'chat.gaps': '知识缺口：',
      'chat.suggestions': '建议：',
      'chat.error': '错误：',
      // Dashboard – reset
      'dashboard.reset.title': '清空知识库',
      'dashboard.reset.desc': '永久删除所有已导入的数据：文本块、实体、图谱及向量库。此操作不可撤销。',
      'dashboard.reset.btn': '清空全部数据',
      'dashboard.reset.done': '知识库已清空。',
      'dashboard.reset.fail': '清空失败，请查看服务器日志。',
      // Graph – toolbar
      'graph.search.placeholder': '筛选节点…',
      'graph.reset': '重置视图',
      'graph.refresh': '刷新',
      // Graph – legend
      'graph.legend.entity': '实体',
      'graph.legend.hyperedge': '超边',
      'graph.legend.unknown': '未知',
      // Graph – details panel
      'graph.details.title': '节点详情',
      'graph.details.type': '类型',
      'graph.details.label': '标签',
      'graph.details.source': '来源',
      'graph.details.degree': '度数',
      'graph.details.connections': '连接',
      'graph.details.none': '无连接',
    },
  };

  let _lang = localStorage.getItem('kc-lang') || 'en';

  window.t = function (key, vars) {
    const dict = TRANSLATIONS[_lang] || TRANSLATIONS.en;
    let str = dict[key] !== undefined ? dict[key] : (TRANSLATIONS.en[key] !== undefined ? TRANSLATIONS.en[key] : key);
    if (vars) {
      Object.keys(vars).forEach(k => { str = str.replace('{' + k + '}', vars[k]); });
    }
    return str;
  };

  window.currentLang = function () { return _lang; };

  window.applyLocale = function (lang) {
    if (lang) { _lang = lang; localStorage.setItem('kc-lang', lang); }
    document.documentElement.lang = _lang === 'zh' ? 'zh-CN' : 'en';

    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    document.querySelectorAll('.lang-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === _lang);
    });

    document.dispatchEvent(new CustomEvent('localechange', { detail: { lang: _lang } }));
  };

  document.addEventListener('DOMContentLoaded', () => applyLocale());
})();
