# SharedFNO RM + flow pipeline

This is the preferred hybrid after keeping the old encoder style but shrinking the latent representation.

```mermaid
flowchart TD
    A["Input fields<br/>(B,V,Ny,Nt)<br/>buoyancy, reduced_shear, log_epsilon"] --> B["Self-supervised corruption<br/>spatial patch mask + variable dropout"]
    B --> C["Variable-wise PatchEmbed<br/>old style"]
    C --> D["Variable-wise ViT<br/>old style"]
    D --> E["Stack variables<br/>(B,V,Npatch,D)<br/>+ variable/availability embedding"]
    E --> F["Cross-variable attention<br/>old style, with var_mask"]
    F --> G["Reshape to patch grid<br/>(B,V,D,Hy,Wt)"]
    G --> H["FNO spectral mixer<br/>on Hy x Wt"]
    H --> I["Conv fusion<br/>(B,V*D,Hy,Wt) -> (B,C,Hy,Wt)"]
    I --> J["Compact latent feature<br/>(B,C,Hy,Wt)"]
    J --> K["RM branch<br/>CNN + global pool + MLP"]
    K --> L["RM prediction"]
    J --> M["Flow branch<br/>upsample latent inside U-Net"]
    M --> N["Velocity prediction"]
    N --> O["Buoyancy reconstruction"]
```

Default shape with `kh_holmboe_dataset_keep_epsilon.h5`, `patch_size=(10,10)`, and `latent_channels=64`:

`input: (B,3,491,200) -> compact latent: (B,64,50,20)`

This keeps the old encoder logic but avoids the old high-resolution encoder output:

`old final feature: (B,64,491,200)`

The compact latent is about 1 percent of the old final feature by element count.
