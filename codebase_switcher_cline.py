import argparse
import subprocess
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

class CodebaseSwitcher:

    STATES = ['preedit', 'beetle', 'sonnet']
    ACTIVE_STATES = ['beetle', 'sonnet']
    EXCLUDE_FILES = ['codebase_switcher_cline.py', '.clineignore']
    EXCLUDE_PATTERNS = [
        'node_modules', '.npm', '.yarn', 'npm-debug.log', 'yarn-debug.log', 'yarn-error.log',
        '.pnpm-debug.log', '.next', '.nuxt', '.vuepress',
        '__pycache__', '.venv', 'venv', '.Python',
        'develop-eggs', 'downloads', 'eggs', '.eggs', 'lib64', 'parts', 'sdist', 'var', 'wheels',
        'target', '.gradle',
        'bin', 'obj',
        '.vscode', '.idea', '.sublime-project', '.sublime-workspace',
        '.DS_Store', '.Spotlight-V100', '.Trashes', 'ehthumbs.db', 'Thumbs.db',
        '.cache', '.temp', '.tmp', 'logs',
    ]
    EXCLUDE_EXTENSIONS = [
        '.pyc', '.pyo', '.class', '.o', '.a', '.lib', '.so', '.dylib', '.dll', '.exe', '.pdb',
        '.log',
        '.tmp', '.temp', '.swp', '.swo', '.bak', '.backup', '.old', '.orig', '.save',
        '.zip', '.tar', '.tar.gz', '.rar', '.7z', '.bz2', '.xz',
        '.deb', '.rpm', '.pkg', '.dmg', '.msi', '.jar', '.war', '.ear',
    ]
    
    GIT_USER = 'Codebase Switcher'
    GIT_EMAIL = 'switcher@codebase.local'
    
    def __init__(self):
        self.current_dir = Path.cwd()
        
    def run_git_command(self, args, capture_output=False):
        """Run git command with cross-platform handling"""
        try:
            cmd = ['git'] + (args if isinstance(args, list) else args.split())
            result = subprocess.run(cmd, capture_output=capture_output, text=True, 
                                  check=not capture_output, cwd=self.current_dir)
            return result.stdout.strip() if capture_output and result.returncode == 0 else (True if not capture_output else None)
        except FileNotFoundError:
            return False if not capture_output else None
        except subprocess.CalledProcessError as e:
            if not capture_output and e.returncode == 128:
                print(f"❌ Git error: {e.stderr.strip() if e.stderr else 'Repository operation failed'}")
            return False if not capture_output else None
    
    def validate_git_repo(self):
        """Check git availability and repo existence"""
        if not shutil.which('git'):
            print("❌ Git not found in PATH.")
            return False
        if not (self.current_dir / '.git').exists():
            print("❌ Not a git repo. Run --init first.")
            return False
        return True
    
    def get_current_branch(self):
        return self.run_git_command(['branch', '--show-current'], capture_output=True) or "main"
    
    def should_exclude_file(self, file_path, zip_name):
        """Check if a file should be excluded from the zip"""
        # Check exact filename matches
        if file_path.name in self.EXCLUDE_FILES or file_path.name == zip_name:
            return True
            
        # Check extension matches
        if file_path.suffix.lower() in self.EXCLUDE_EXTENSIONS:
            return True
            
        # Check if any part of the path matches exclude patterns
        path_parts = file_path.parts
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in path_parts or file_path.name == pattern:
                return True
                
        # Check for files that start with exclude patterns (like .DS_Store variants)
        for pattern in self.EXCLUDE_PATTERNS:
            if file_path.name.startswith(pattern):
                return True
                
        return False
    
    def branch_exists(self, branch):
        result = self.run_git_command(['branch', '--list', branch], capture_output=True)
        return branch in result if result else False
    
    def has_uncommitted_changes(self):
        """Check if there are uncommitted changes"""
        return bool(self.run_git_command(['status', '--porcelain'], capture_output=True))
    
    def auto_commit_changes(self, message_suffix=""):
        """Auto-commit any pending changes"""
        if not self.has_uncommitted_changes():
            return True
            
        current = self.get_current_branch()
        print(f"💾 Auto-committing changes on {current}{message_suffix}")
        
        if not self.run_git_command(['add', '.']):
            print("❌ Failed to stage changes")
            return False
        if not self.run_git_command(['commit', '-m', f'Auto-commit on {current}{message_suffix}']):
            print("❌ Failed to commit changes")
            return False
        return True
    
    def setup_git_config(self):
        """Set up local git configuration"""
        print("🔧 Setting up local git configuration...")
        if not self.run_git_command(['config', '--local', 'user.name', self.GIT_USER]):
            print("⚠️  Warning: Could not set local git user.name")
        if not self.run_git_command(['config', '--local', 'user.email', self.GIT_EMAIL]):
            print("⚠️  Warning: Could not set local git user.email")
    
    def create_initial_files(self):
        """Create README, .clineignore, and .gitignore files"""
        try:
            readme_path = self.current_dir / 'README.md'
            readme_path.write_text(
                f"# Model Comparison Project\n\nProject Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"This project contains model comparison results.\n\n"
                f"- `preedit` branch: Original baseline codebase\n"
                f"- `beetle` branch: Beetle model's response\n"
                f"- `sonnet` branch: Sonnet model's response\n",
                encoding='utf-8'
            )
        except (OSError, UnicodeError) as e:
            print(f"⚠️  Warning: Could not create README.md: {e}")
        
        try:
            self._update_clineignore()
        except (OSError, UnicodeError) as e:
            print(f"⚠️  Warning: Could not create/update .clineignore: {e}")
        
        try:
            self._create_gitignore()
        except (OSError, UnicodeError) as e:
            print(f"⚠️  Warning: Could not create .gitignore: {e}")
        
    def _update_clineignore(self):
        """Helper to update .clineignore file"""
        clineignore_path = self.current_dir / '.clineignore'
        switcher_entry = "codebase_switcher_cline.py"
        
        if clineignore_path.exists():
            existing_content = clineignore_path.read_text(encoding='utf-8')
            if switcher_entry not in existing_content:
                new_content = existing_content.rstrip() + "\n\n# Ignore codebase switcher script\n"
                new_content += switcher_entry + "\n"
                clineignore_path.write_text(new_content, encoding='utf-8')
        else:
            clineignore_path.write_text(
                "# Ignore codebase switcher script\n"
                "codebase_switcher_cline.py\n",
                encoding='utf-8'
            )
    
    def _create_gitignore(self):
        """Helper to create .gitignore file for excluding large/generated files"""
        gitignore_path = self.current_dir / '.gitignore'
        
        # Convert patterns and extensions to gitignore format
        gitignore_patterns = []
        
        # Add directory patterns
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern.startswith('.'):
                gitignore_patterns.append(f"{pattern}")
                gitignore_patterns.append(f"**/{pattern}")
            else:
                gitignore_patterns.append(f"{pattern}/")
                gitignore_patterns.append(f"**/{pattern}/")
        
        # Add extension patterns
        gitignore_patterns.extend(f"*{ext}" for ext in self.EXCLUDE_EXTENSIONS)
        
        gitignore_content = f"""# Auto-generated by codebase switcher
# Excludes large files and build artifacts to keep branches clean

{chr(10).join(gitignore_patterns)}
"""
        
        gitignore_path.write_text(gitignore_content, encoding='utf-8')
    
    def create_branches(self):
        """Create and set up all required branches"""
        if not self.branch_exists('preedit'):
            print("📝 Creating preedit branch (baseline)")
            if not self.run_git_command(['checkout', '-b', 'preedit']):
                print("❌ Failed to create preedit branch")
                return False
        else:
            if not self.run_git_command(['checkout', 'preedit']):
                print("❌ Failed to switch to preedit branch")
                return False
        
        for state in ['beetle', 'sonnet']:
            if not self.branch_exists(state):
                print(f"📝 Creating {state} branch from preedit baseline")
                if not self.run_git_command(['checkout', '-b', state]):
                    print(f"❌ Failed to create {state} branch")
                    return False
                if not self.run_git_command(['checkout', 'preedit']):
                    print("❌ Failed to return to preedit branch")
                    return False
        
        return True
    
    def initialize(self):
        """Initialize git repo for model comparison"""
        if not shutil.which('git'):
            print("❌ Git not found in PATH.")
            return False
            
        if not (self.current_dir / '.git').exists():
            print("🔧 Initializing git repository...")
            if not self.run_git_command(['init']):
                return False
            self.setup_git_config()
        
        if not self.run_git_command(['log', '--oneline', '-1'], capture_output=True):
            print("📝 Creating initial commit...")
            self.create_initial_files()
            
            if not self.run_git_command(['add', '.']):
                print("❌ Failed to stage files")
                return False
            if not self.run_git_command(['commit', '-m', 'Initial commit - baseline for model comparison']):
                print("❌ Failed to create initial commit")
                return False
        
        if not self.auto_commit_changes(" before branch setup"):
            return False
        
        if not self.create_branches():
            return False
            
        print(f"✅ Initialized model comparison environment")
        print(f"📍 Ready for experimentation")
        print(f"")
        print(f"🎯 EXPERIMENTATION WORKFLOW:")
        print(f"1. Work on PREEDIT branch with Beetle model")
        print(f"2a. If Beetle response is TOO GOOD:")
        print(f"    → --accept-baseline (make it new baseline, continue experimenting)")
        print(f"2b. If Beetle response is OKAY/WORSE:")
        print(f"    → --beetle (save & start comparison)")
        print(f"3. Work with Sonnet model: --sonnet")
        print(f"4. Package results: --zip")
        print(f"")
        print(f"💡 Use --rollback anytime to discard changes and try again")
        return True
    
    def switch_state(self, state):
        """Switch to specified model branch with save and reset logic"""
        if state not in self.ACTIVE_STATES:
            print(f"❌ Invalid state. Available: {', '.join(self.ACTIVE_STATES)}")
            return False
        
        if not self.validate_git_repo():
            return False
        
        if not self.branch_exists(state):
            print(f"❌ State '{state}' doesn't exist. Run --init first.")
            return False
        
        current = self.get_current_branch()
        
        # If on preedit with changes, save them to the state branch and reset preedit
        if current == 'preedit' and self.has_uncommitted_changes():
            return self.save_and_reset_preedit(state)
        
        # If already on the target state
        if current == state:
            print(f"✅ Already on {state} branch")
            print(f"🤖 Ready for {state.title()} model work")
            return True
        
        # Handle other branch switches (with auto-commit if needed)
        if self.has_uncommitted_changes():
            if not self.auto_commit_changes(f" - {current} model work"):
                return False
        
        print(f"🔄 Switching to {state} branch...")
        if self.run_git_command(['checkout', state]):
            print(f"✅ Now on {state} branch")
            print(f"🤖 Ready for {state.title()} model work")
            return True
        
        print(f"❌ Failed to switch to {state}")
        return False
    
    def show_status(self):
        """Show current project status"""
        if not self.validate_git_repo():
            return
        
        current = self.get_current_branch()
        has_changes = self.has_uncommitted_changes()
        
        print(f"📍 Current branch: {current}")
        print(f"🤖 Available models: {', '.join(self.ACTIVE_STATES)}")
        
        if current == 'preedit':
            if has_changes:
                print("WARNING: You have uncommitted changes on preedit")
            else:
                print("✨ Clean preedit baseline - ready for experimentation")
                print("   💡 Work freely with Beetle until you get 'okay or worse' quality")
        elif current in self.ACTIVE_STATES:
            print(f"🤖 On {current} branch - working with {current.title()}")
            if has_changes:
                print("⚡ You have uncommitted work in progress")
        
        print(f"\n🔄 Git status:")
        status_result = self.run_git_command(['status', '--short'], capture_output=True)
        if status_result:
            print(status_result)
        else:
            print("No changes")
            
        print(f"\n📚 Recent commits:")
        log_result = self.run_git_command(['log', '--oneline', '-3'], capture_output=True)
        if log_result:
            print(log_result)
        else:
            print("No commits found")
    
    def show_version(self):
        """Show version"""
        print("v.0628_1410_cline")
    
    def cleanup_switcher_branches(self):
        """Remove switcher branches, keep files"""
        print("🧹 Cleaning up switcher branches...")
        
        current = self.get_current_branch()
        safe_branch = next((b for b in ['main', 'master'] if self.branch_exists(b)), None)
        
        if not safe_branch:
            print("🔄 Creating main branch...")
            if not self.run_git_command(['checkout', '-b', 'main']):
                print("❌ Failed to create main branch")
                return False
        elif current in self.STATES:
            print(f"🔄 Switching to {safe_branch}...")
            if not self.run_git_command(['checkout', safe_branch]):
                print(f"❌ Failed to switch to {safe_branch}")
                return False
        
        for state in self.STATES:
            if self.branch_exists(state):
                print(f"🗑️  Deleting {state} branch...")
                self.run_git_command(['branch', '-D', state])
        
        print("✅ Cleanup complete - switcher branches removed")
        return True
    
    def verify_branches_different(self):
        """Verify that model branches have different results"""
        print("🔍 Verifying model results are different...")
        
        for state in self.ACTIVE_STATES:
            if not self.branch_exists(state):
                print(f"❌ Branch '{state}' doesn't exist. Complete model work first.")
                return False
        
        try:
            result = subprocess.run(['git', 'diff', '--quiet', 'beetle', 'sonnet'], 
                                  cwd=self.current_dir, capture_output=True, check=False)
            if result.returncode == 0:
                print("⚠️  Warning: Beetle and Sonnet branches are identical")
                print("   This suggests the models gave the same response")
                return True  # Still allow zip creation
        except FileNotFoundError:
            print("❌ Git not found")
            return False
        
        print("✅ Model responses are different - good comparison data")
        return True
    
    def create_zip(self):
        """Create model comparison results zip"""
        if not self.validate_git_repo():
            return False
        
        # Auto-commit any pending changes
        current = self.get_current_branch()
        if self.has_uncommitted_changes():
            if not self.auto_commit_changes(f" - final {current} work"):
                return False
        
        if not self.verify_branches_different():
            return False
        
        # Switch to preedit so extracted zip starts from baseline
        print("🔄 Switching to preedit baseline for zip creation...")
        if not self.run_git_command(['checkout', 'preedit']):
            print("❌ Failed to switch to preedit")
            return False
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f"model_comparison_{self.current_dir.name}_{timestamp}.zip"
        zip_path = self.current_dir / zip_name
        
        print(f"📦 Creating model comparison results: {zip_name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.current_dir.rglob('*'):
                    if file_path.is_file() and not self.should_exclude_file(file_path, zip_name):
                        try:
                            arc_path = file_path.relative_to(self.current_dir).as_posix()
                            zipf.write(file_path, arc_path)
                        except (OSError, ValueError) as e:
                            print(f"⚠️  Skipping {file_path.name}: {e}")
                            continue
            
            file_size_kb = zip_path.stat().st_size / 1024
            print(f"✅ Created {zip_name} ({file_size_kb:.1f} KB)")
            print("📋 Contains: Model comparison results with git history")
            print("📊 Ready for analysis!")
            
            self.cleanup_switcher_branches()
            return True
            
        except Exception as e:
            print(f"❌ Error creating zip: {e}")
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except OSError:
                    pass
            return False

    def accept_baseline(self):
        """Accept current preedit state as new baseline for all branches"""
        current = self.get_current_branch()
        if current != 'preedit':
            print("❌ Must be on preedit branch to accept baseline")
            return False
            
        # Commit any pending changes
        if self.has_uncommitted_changes():
            if not self.auto_commit_changes(" - accepted as new baseline"):
                return False
        
        print("✅ Accepting current preedit state as new baseline...")
        print("🔄 Updating all branches to match current preedit state...")
        
        # Update beetle and sonnet branches to match current preedit
        for state in ['beetle', 'sonnet']:
            if not self.run_git_command(['checkout', state]):
                print(f"❌ Failed to switch to {state}")
                return False
            if not self.run_git_command(['reset', '--hard', 'preedit']):
                print(f"❌ Failed to reset {state} to preedit")
                return False
            print(f"✅ Updated {state} branch to new baseline")
        
        # Return to preedit
        if not self.run_git_command(['checkout', 'preedit']):
            print("❌ Failed to return to preedit")
            return False
            
        print("🎯 New baseline established! All branches now match current state.")
        print("📍 Continue experimenting on this new baseline.")
        return True

    def rollback_preedit(self):
        """Rollback preedit to clean baseline state (discard current changes)"""
        current = self.get_current_branch()
        if current != 'preedit':
            print("❌ Must be on preedit branch to rollback")
            return False
            
        if not self.has_uncommitted_changes():
            print("✅ No changes to rollback - already at clean baseline")
            return True
        
        print("🔄 Rolling back preedit to clean baseline state...")
        print("⚠️  This will discard all current changes!")
        
        # Reset to clean state (discard all changes)
        if not self.run_git_command(['reset', '--hard']):
            print("❌ Failed to reset preedit")
            return False
        
        # Also clean any untracked files
        if not self.run_git_command(['clean', '-fd']):
            print("⚠️  Warning: Could not clean untracked files")
        
        print("✅ Rolled back to clean baseline state")
        print("📍 Ready for fresh experimentation on clean preedit")
        return True

    def save_and_reset_preedit(self, model_name):
        """Save preedit changes to model branch and reset preedit to clean state"""
        current = self.get_current_branch()
        if current != 'preedit':
            print(f"❌ Must be on preedit branch to save {model_name} response")
            return False
            
        if not self.has_uncommitted_changes():
            print("❌ No changes to save")
            return False
        
        print(f"💾 Saving {model_name} response to {model_name} branch...")
        
        # Commit changes on preedit first
        if not self.auto_commit_changes(f" - {model_name} response"):
            return False
        
        # Switch to model branch and merge the changes
        if not self.run_git_command(['checkout', model_name]):
            print(f"❌ Failed to switch to {model_name} branch")
            return False
            
        if not self.run_git_command(['merge', 'preedit']):
            print(f"❌ Failed to merge preedit changes into {model_name}")
            return False
        
        print(f"✅ Saved {model_name} response to {model_name} branch")
        
        # Return to preedit and reset to clean state (before the model's changes)
        if not self.run_git_command(['checkout', 'preedit']):
            print("❌ Failed to return to preedit branch")
            return False
            
        if not self.run_git_command(['reset', '--hard', 'HEAD~1']):
            print("❌ Failed to reset preedit to clean state")
            return False
            
        print("🔄 Reset preedit branch to clean baseline state")
        print(f"📍 Ready for {model_name} vs other model comparison")
        return True

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Tool - Experimentation & Comparison')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--init', action='store_true', help='Initialize model comparison environment')
    group.add_argument('-2', '--beetle', action='store_true', help='Save preedit to beetle branch and reset (if on preedit), or switch to beetle') 
    group.add_argument('-3', '--sonnet', action='store_true', help='Save preedit to sonnet branch (if on preedit), or switch to sonnet')
    group.add_argument('-a', '--accept-baseline', action='store_true', help='Accept current preedit as new baseline for all branches')
    group.add_argument('-r', '--rollback', action='store_true', help='Rollback preedit to clean baseline (discard changes)')
    group.add_argument('-s', '--status', action='store_true', help='Show project status with guidance')
    group.add_argument('-z', '--zip', action='store_true', help='Create model comparison results zip')
    group.add_argument('-v', '--version', action='store_true', help='Show version')
    
    args = parser.parse_args()
    switcher = CodebaseSwitcher()
    
    if args.init:
        switcher.initialize()
    elif args.beetle:
        switcher.switch_state('beetle')
    elif args.sonnet:
        switcher.switch_state('sonnet')
    elif args.accept_baseline:
        switcher.accept_baseline()
    elif args.rollback:
        switcher.rollback_preedit()
    elif args.status:
        switcher.show_status()
    elif args.zip:
        switcher.create_zip()
    elif args.version:
        switcher.show_version()

if __name__ == '__main__':
    main() 