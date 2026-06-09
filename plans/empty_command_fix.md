# Plan: Corriger la boucle infinie sur commandes vides

## Problème

Quand l'IA appelle `execute_command` avec une commande vide (`""`), le code :
1. Vérifie si c'est dangereux → non (chaîne vide)
2. Exécute `subprocess.run("", ...)` → échec silencieux ou résultat vide
3. L'IA ne comprend pas pourquoi ça n'a pas marché et réessaie en boucle
4. Atteint `max_depth=15` et s'arrête

## Solution

Ajouter une validation explicite : si la commande est vide ou ne contient que des espaces, retourner immédiatement un message d'erreur clair à l'IA.

## Changement dans [`nyx/agent.py`](nyx/agent.py:456-474)

```python
if name == "execute_command":
    import subprocess
    command = args.get("command", "").strip()
    timeout = args.get("timeout", 30)

    # --- NOUVEAU : validation commande vide ---
    if not command:
        logger.warning("Empty command received from AI")
        return (
            "[ERROR] Empty command received. You must provide a valid shell command "
            "in the 'command' parameter. For example: 'ls -la', 'cat file.txt', "
            "'python3 script.py', etc."
        )

    # If the command matches dangerous patterns, request user approval
    if self._is_dangerous_command(command):
        ...
```

## Fichier à modifier

| Fichier | Modification |
|---------|-------------|
| [`nyx/agent.py`](nyx/agent.py) | Ajouter validation commande vide avant les vérifications de sécurité |

## Test à ajouter dans [`tests/test_agent.py`](tests/test_agent.py)

```python
def test_empty_command_returns_error(self):
    """An empty command should return a clear error message."""
    tc = ToolCall(id="1", name="execute_command", arguments={"command": ""})
    result = self.agent._execute_tool(tc)
    assert "ERROR" in result
    assert "Empty command" in result