document.addEventListener('DOMContentLoaded', () => {
    const runsList = document.getElementById('runsList');
    const newRunBtn = document.getElementById('newRunBtn');
    
    const newRunView = document.getElementById('newRunView');
    const chatView = document.getElementById('chatView');
    
    const ingestForm = document.getElementById('ingestForm');
    const chatHistory = document.getElementById('chatHistory');
    const chatTitle = document.getElementById('chatTitle');
    const reportContainer = document.getElementById('reportContainer');
    
    const contextMenu = document.getElementById('contextMenu');
    const amendOption = document.getElementById('amendOption');
    const amendModal = document.getElementById('amendModal');
    const amendForm = document.getElementById('amendForm');
    const cancelAmendBtn = document.getElementById('cancelAmendBtn');
    const amendRunIdSpan = document.getElementById('amendRunId');

    let currentEventSource = null;
    let contextRunId = null;

    // Load initial state
    init();

    async function init() {
        await loadRuns();
        await checkActiveRun();
    }

    async function checkActiveRun() {
        try {
            const res = await fetch('/api/active_run');
            const data = await res.json();
            
            if (data.status === 'running') {
                showView(chatView);
                chatTitle.textContent = "Investigation In Progress...";
                chatHistory.innerHTML = '';
                reportContainer.innerHTML = '';
                reportContainer.classList.add('hidden');
                
                if (data.thoughts && data.thoughts.length > 0) {
                    data.thoughts.forEach(t => appendMessage('thought', t));
                }
                
                connectStream();
            }
        } catch (e) {
            console.error("Failed to check active run", e);
        }
    }

    async function loadRuns() {
        try {
            const res = await fetch('/api/runs');
            const data = await res.json();
            runsList.innerHTML = '';
            data.runs.forEach(run => {
                const div = document.createElement('div');
                div.className = 'run-item';
                div.innerHTML = `
                    <div class="run-id">${run.id}</div>
                    <div class="timestamp">${new Date(run.timestamp * 1000).toLocaleString()}</div>
                `;
                
                div.addEventListener('click', () => loadRun(run.id));
                div.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    contextRunId = run.id;
                    contextMenu.style.left = `${e.pageX}px`;
                    contextMenu.style.top = `${e.pageY}px`;
                    contextMenu.classList.remove('hidden');
                });
                runsList.appendChild(div);
            });
        } catch (e) {
            console.error('Failed to load runs', e);
        }
    }

    document.addEventListener('click', () => {
        contextMenu.classList.add('hidden');
    });

    amendOption.addEventListener('click', () => {
        if (!contextRunId) return;
        amendRunIdSpan.textContent = contextRunId;
        amendModal.classList.remove('hidden');
    });

    cancelAmendBtn.addEventListener('click', () => {
        amendModal.classList.add('hidden');
        amendForm.reset();
    });

    newRunBtn.addEventListener('click', () => {
        showView(newRunView);
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
    });

    async function loadRun(runId) {
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
        showView(chatView);
        chatTitle.textContent = `Run: ${runId}`;
        chatHistory.innerHTML = '';
        reportContainer.innerHTML = '';
        reportContainer.classList.add('hidden');

        try {
            const res = await fetch(`/api/runs/${runId}`);
            const data = await res.json();
            
            if (data.thoughts && data.thoughts.length > 0) {
                data.thoughts.forEach(thought => {
                    appendMessage('thought', thought);
                });
            }

            if (data.report) {
                reportContainer.innerHTML = marked.parse(data.report);
                reportContainer.classList.remove('hidden');
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            } else if (!data.thoughts || data.thoughts.length === 0) {
                appendMessage('error', 'Report or investigation steps not found for this run.');
            }
        } catch (e) {
            console.error(e);
        }
    }

    function showView(view) {
        newRunView.classList.add('hidden');
        chatView.classList.add('hidden');
        view.classList.remove('hidden');
    }

    ingestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fd = new FormData();
        const files = ['baseline_app', 'baseline_container', 'cluster_app', 'cluster_container', 'incident_traces', 'incident_logs'];
        let hasFiles = false;
        files.forEach(fid => {
            const input = document.getElementById('f_' + fid);
            if (input.files.length > 0) {
                fd.append(fid, input.files[0]);
                hasFiles = true;
            }
        });

        const btn = e.target.querySelector('button[type="submit"]');
        btn.textContent = "Processing...";
        btn.disabled = true;

        if (hasFiles) {
            try {
                await fetch('/api/ingest', { method: 'POST', body: fd });
            } catch (err) {
                console.error(err);
                alert("Ingest failed");
                btn.textContent = "Ingest & Run";
                btn.disabled = false;
                return;
            }
        }

        const incidentTs = document.getElementById('incident_ts').value;
        const hyp = document.getElementById('hypotheses').value;
        const maxSteps = document.getElementById('max_steps').value;
        
        const startRes = await fetch(`/api/start_run?incident_ts=${encodeURIComponent(incidentTs)}&hypotheses=${encodeURIComponent(hyp)}&max_steps=${encodeURIComponent(maxSteps)}`);
        const startData = await startRes.json();
        
        btn.textContent = "Ingest & Run";
        btn.disabled = false;

        if (startData.status === 'started' || startData.status === 'already_running') {
            showView(chatView);
            chatTitle.textContent = "Investigation In Progress...";
            chatHistory.innerHTML = '';
            reportContainer.innerHTML = '';
            reportContainer.classList.add('hidden');
            connectStream();
        }
    });

    amendForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = document.getElementById('amend_question').value;
        const s = document.getElementById('amend_steps').value;
        amendModal.classList.add('hidden');
        amendForm.reset();
        
        const startRes = await fetch(`/api/start_amend?bundle=${encodeURIComponent(contextRunId)}&question=${encodeURIComponent(q)}&max_steps=${encodeURIComponent(s)}`);
        const startData = await startRes.json();

        if (startData.status === 'started' || startData.status === 'already_running') {
            showView(chatView);
            chatTitle.textContent = "Investigation In Progress...";
            chatHistory.innerHTML = '';
            reportContainer.innerHTML = '';
            reportContainer.classList.add('hidden');
            connectStream();
        }
    });

    function connectStream() {
        if (currentEventSource) {
            currentEventSource.close();
        }
        
        currentEventSource = new EventSource('/api/stream');
        
        currentEventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            
            if (data.type === 'thought') {
                appendMessage('thought', data.content);
            } else if (data.type === 'hypothesis') {
                appendMessage('thought', `_Hypothesis written: \`${data.id}\`_`);
            } else if (data.type === 'error') {
                appendMessage('error', `Error: ${data.content}`);
                currentEventSource.close();
            } else if (data.type === 'done') {
                chatTitle.textContent = `Completed: ${data.result.run_id}`;
                currentEventSource.close();
                loadRuns();
                
                setTimeout(() => loadRun(data.result.run_id), 500);
            }
        };
        
        currentEventSource.onerror = () => {
            currentEventSource.close();
        };
    }

    function appendMessage(type, content) {
        if (!content) return;
        const div = document.createElement('div');
        div.className = `message ${type}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = marked.parse(content);
        
        div.appendChild(bubble);
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
