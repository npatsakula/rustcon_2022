{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-compat = { url = "github:edolstra/flake-compat"; flake = false; };
    crane = { url = "github:ipetkov/crane"; inputs.nixpkgs.follows = "nixpkgs"; };
  };
  outputs = { self, nixpkgs, utils, rust-overlay, crane, flake-compat }:
    utils.lib.eachSystem [ utils.lib.system.x86_64-linux ] (system:
      let
        pkgs = (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
          config.allowUnfree = true;
        }).extend(self: super: {
          patched_rust = self.rust-bin.stable.latest.default;
        });

      fonts = pkgs.makeFontsConf {
        fontDirectories = with pkgs; [
          corefonts
        ];
      };

      tex = (pkgs.texlive.combine {
        inherit (pkgs.texlive) scheme-medium
          babel-russian pdftex latexmk
          biblatex xltxtra biber csquotes hyperref
          ;
      });

      mkShell = pkgs.mkShell.override {
        stdenv = pkgs.llvmPackages_14.stdenv;
      };

      crane' = (crane.mkLib pkgs).overrideToolchain pkgs.patched_rust;
      nativeBuildInputs = with pkgs; [ libffi zlib.dev ncurses libxml2.dev ];
      in {
        devShells.default = mkShell {
          buildInputs = with pkgs; [ patched_rust ] ++ nativeBuildInputs;
          LLVM_SYS_140_PREFIX = "${pkgs.llvmPackages_14.llvm.dev}/";
        };

        devShells.latex = mkShell {
          buildInputs = [ tex ];
          FONTCONFIG_FILE="${fonts}";
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