# RTX VFX : les trois nouveaux nodes

## RTX Artifact Reduction

Identifiant : `CMDR_RTXArtifactReduction`.

| Entrée | Défaut | Fonction |
|---|---|---|
| images | — | IMAGE, batch RGB ou RGBA (alpha ignoré) |
| strength | LOW | LOW, MED, HIGH, ULTRA selon le SDK installé |
| output_dtype | float16 | float16 ou float32 |
| keep_loaded | true | Conserver le moteur GPU entre les exécutions |
| audio | optionnel | AUDIO retransmis sans modification |

Sorties : `images` (IMAGE) et `audio` (AUDIO, `None` si absent).

**720×576 entre → 720×576 sort**, y compris pour les dimensions impaires.
Aucun paramètre scale, width, height ou align. Une sortie du SDK qui aurait une
autre taille provoque une erreur : aucun redimensionnement de secours.

Le backend reprend `VideoSuperRes` du node existant en utilisant exclusivement
`DENOISE_LOW`, `DENOISE_MEDIUM`, `DENOISE_HIGH`, `DENOISE_ULTRA`.
Ces modes NVIDIA enlèvent bruit et artefacts de compression à taille identique.
Il ne s'agit pas de l'ancien filtre natif `ArtifactReduction`, qui n'a que deux
modes et ne correspond pas aux quatre niveaux demandés. Un niveau plus élevé
peut lisser les textures fines. Un SDK trop ancien produit une erreur explicite.

## RTX AI Green Screen

Identifiant : `CMDR_RTXAIGreenScreen`.

Entrées : `images`, `mode`, `temporal` (true), `output_dtype` (float16),
`keep_loaded` (true).

Les modes suivent les valeurs NVIDIA :

| Mode | Valeur SDK |
|---|---|
| QUALITY (chairs foreground) — défaut | 0 |
| PERFORMANCE (chairs foreground) | 1 |
| QUALITY (chairs background) | 2 |
| PERFORMANCE (chairs background) | 3 |

Sorties :

- `foreground` : IMAGE RGB du sujet sur fond noir, calculé par image × masque.
- `mask` : MASK float32 `[N,H,W]`, entre 0 et 1 ; blanc = premier plan,
  noir = arrière-plan. Ce n'est pas le masque inversé d'un chargeur d'alpha.

Le SDK exige au moins **512 pixels de large et 288 de haut**. Les entrées trop
petites sont refusées sans redimensionnement. La résolution du masque est celle
de l'image. Le modèle cible surtout une personne face caméra, pas une segmentation
universelle d'objets.

`temporal=true` traite les frames comme une séquence ordonnée. `false` réinitialise
l'état entre chaque image indépendante. Dans les deux cas, l'état est remis à
zéro au début de chaque exécution, même si le moteur reste chargé. La continuité
temporelle ne traverse donc pas deux batches ComfyUI distincts.

## RTX Background Blur

Identifiant : `CMDR_RTXBackgroundBlur`.

Entrées : `images`, `mask`, `strength` (0.5, entre 0 et 1), `output_dtype`
(float16), `keep_loaded` (true). Sortie : `images` (IMAGE).

Brancher **l'image originale**, ainsi que le masque de Green Screen :

```text
IMAGE originale ──┬── RTX AI Green Screen ── mask ──┐
                  └─────────────────────────────── RTX Background Blur → IMAGE
```

Le blanc protège le sujet, le noir désigne la zone à flouter. Les dimensions du
masque doivent correspondre exactement. Un masque `[H,W]` ou `[1,H,W]` est réutilisé
pour tout le batch ; sinon il faut exactement un masque par image.

Green Screen et Background Blur utilisent réellement l'API native NVIDIA, avec
conversion GPU RGB float → BGR uint8 et masque alpha uint8, conformément au SDK.
Le réglage output_dtype change le type de sortie, pas la précision native 8 bits.

## Installation

1. Installer ce pack dans `ComfyUI/custom_nodes/COMFYUI_helpers_nodes` puis
   redémarrer ComfyUI. Les trois nodes sont dans `Helpers 🧰`.
2. Pour Artifact Reduction, utiliser le même environnement `nvidia-vfx` que le
   RTX upscaler, avec une version proposant les modes `DENOISE_*`.
3. Pour les deux autres nodes, installer le **SDK VFX natif** et les features
   NVIDIA `nvvfxgreenscreen` et `nvvfxbackgroundblur`, avec leurs dépendances et
   modèles. Le seul paquet Python `nvidia-vfx` ne suffit pas.
4. Si les bibliothèques ne sont pas découvertes automatiquement, définir avant
   de lancer ComfyUI :

```powershell
$env:NV_VIDEO_EFFECTS_PATH = 'D:\NVIDIA\VFX\bin'
$env:NVVFX_MODEL_DIR = 'D:\NVIDIA\VFX\models'
```

Adapter ces chemins à l'installation réelle. Le premier doit contenir
`NVVideoEffects.dll` (ou `libNVVideoEffects.so` sous Linux), directement ou dans
`bin`/`lib`. Les dépendances du SDK doivent également être accessibles au chargeur
du système. Sous Windows, le dossier historique
`C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects` est aussi recherché.
Sous Linux, configurer `LD_LIBRARY_PATH` avant de démarrer ComfyUI si nécessaire.

Aucune installation ni aucun téléchargement automatique à l'import du pack.
Les nodes restent enregistrés sans SDK et expliquent le problème à l'exécution.

## Cache et validation

Traitement GPU frame par frame ; le batch de sortie reste en RAM. Les effets
natifs conservent au plus un moteur chacun, remplacé si GPU, dimensions ou
paramètres changent. Artifact Reduction a un cache séparé du RTX upscaler.
`keep_loaded=false` ferme la session après traitement. Une erreur ou interruption
ferme également la session. Les appels aux nouveaux moteurs sont sérialisés.

Tests CPU (PyTorch requis, ComfyUI et SDK remplacés par des doubles de test) :

```text
python tests/test_rtx_vfx_nodes.py
```

Ces tests vérifient dimensions, types, audio, masques, interruption et cache.
**L'inférence réelle CUDA/NVIDIA doit encore être validée sur une machine avec
les features installées.** Aucune qualité visuelle ou compatibilité binaire de
l'installation locale ne peut être déduite des tests simulés.

## Références NVIDIA

- [API Python et modes DENOISE](https://docs.nvidia.com/maxine/vfx-python/latest/api.html)
- [AI Green Screen](https://docs.nvidia.com/maxine/vfx/latest/Filters/AIGreenScreen.html)
- [Background Blur](https://docs.nvidia.com/maxine/vfx/latest/Filters/BackgroundBlur.html)
- [Exemple natif AigsEffectApp](https://github.com/NVIDIA-Maxine/VFX-SDK-Samples/blob/main/apps/AigsEffectApp/AigsEffectApp.cpp)
- [En-têtes SDK](https://github.com/NVIDIA-Maxine/Maxine-VFX-SDK/tree/main/nvvfx/include)
