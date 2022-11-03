{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-compat = { url = "github:edolstra/flake-compat"; flake = false; };
  };

  outputs = { self, nixpkgs, utils, rust-overlay, flake-compat }:
    utils.lib.eachSystem [ utils.lib.system.x86_64-linux ] (system:
      let
        name = "evac";

        pkgs = (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
          config.allowUnfree = true;
        }).extend(self: super: rec {
          rustPatched = self.rust-bin.stable.latest.default;
          rustfmt = rustPatched;
          rustc = rustPatched;
          cargo = rustPatched;
          rust-analyzer = rustPatched;
        }) // {
          stdenv = pkgs.llvmPackages_14.stdenv;
        };

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

      in rec {
        devShell = pkgs.mkShell rec {
          buildInputs = with pkgs; [
            rustc cargo rust-analyzer
            valgrind gperftools pprof graphviz
            tex
          ];
          FONTCONFIG_FILE="${fonts}";
        };
      });
}