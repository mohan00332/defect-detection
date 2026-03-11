import { spawn } from 'child_process';

const children = [];

function start(cmd, args, label) {
  const child = spawn(cmd, args, { stdio: 'inherit', shell: false });
  child.on('exit', (code) => {
    if (code && code !== 0) {
      console.error(`${label} exited with code ${code}`);
    }
  });
  children.push(child);
}

function shutdown(signal = 'SIGINT') {
  for (const child of children) {
    try {
      child.kill(signal);
    } catch {
      // ignore
    }
  }
}

process.on('SIGINT', () => {
  shutdown('SIGINT');
  process.exit(0);
});

process.on('SIGTERM', () => {
  shutdown('SIGTERM');
  process.exit(0);
});

start('python', ['app.py'], 'Python backend');
start('node', ['server.js'], 'Node UI');
