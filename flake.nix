{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-compat = { url = "github:edolstra/flake-compat"; flake = false; };
    crane = { url = "github:ipetkov/crane"; inputs.nixpkgs.follows = "nixpkgs"; };
  };
  outputs = { self, nixpkgs, utils, rust-overlay, crane, flake-compat }:
    utils.lib.eachSystem (with utils.lib.system; [ x86_64-linux aarch64-darwin ]) (system:
      let
        pkgs = (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
          config.allowUnfree = true;
        }).extend (self: super: {
          patched_rust = self.rust-bin.stable.latest.default;
        });

        mkShell = pkgs.mkShell.override {
          stdenv = pkgs.llvmPackages_14.stdenv;
        };

        crane' = (crane.mkLib pkgs).overrideToolchain pkgs.patched_rust;
        nativeBuildInputs = with pkgs; [ libffi zlib.dev ncurses.dev libxml2.dev ];
      in
      {
        devShells =
          rec {
            rust = mkShell {
              buildInputs = with pkgs; [ patched_rust ] ++ nativeBuildInputs;
              LLVM_SYS_140_PREFIX = "${pkgs.llvmPackages_14.llvm.dev}/";
            };

            latex = mkShell {
              buildInputs = [ pkgs.tectonic pkgs.python3Packages.pygments ];
            };

            default = mkShell {
              buildInputs = [ pkgs.rnix-lsp ] ++ rust.buildInputs ++ latex.buildInputs;
            };
          };

        packages = rec {
          default = crane'.buildPackage {
            src = pkgs.lib.cleanSource ./.;
            inherit nativeBuildInputs;
            LLVM_SYS_140_PREFIX = "${pkgs.llvmPackages_14.llvm.dev}/";
          };

          container = pkgs.dockerTools.buildLayeredImage {
            name = default.pname;
            tag = "${default.version}-${self.sourceInfo.shortRef or "dirty"}";
            contents = [ default ];
            created = "now";
          };
        };
      });
}
