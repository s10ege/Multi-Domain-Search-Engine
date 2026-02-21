import Link from "next/link";
import styles from "./sign-in.module.css";

export default function SignInPage() {
  return (
    <div className={styles.page}>
      <div className={styles.glow} aria-hidden="true" />
      <main className={styles.card}>
        <div className={styles.brand}>SemantikML</div>
        <h1 className={styles.title}>Sign in</h1>
        <p className={styles.subtitle}>
          Continue with your workspace or use email.
        </p>

        <div className={styles.actions}>
          <button className={styles.primaryButton} type="button">
            Continue with workspace
          </button>
          <button className={styles.ghostButton} type="button">
            Use email instead
          </button>
        </div>

        <div className={styles.divider}>
          <span>or</span>
        </div>

        <form className={styles.form}>
          <label className={styles.label}>
            Email
            <input
              className={styles.input}
              type="email"
              placeholder="you@semantikml.ai"
            />
          </label>
          <label className={styles.label}>
            Password
            <input
              className={styles.input}
              type="password"
              placeholder="********"
            />
          </label>
          <button className={styles.submitButton} type="submit">
            Sign in
          </button>
        </form>

        <div className={styles.footer}>
          <span>New here?</span>
          <a className={styles.link} href="#">
            Create an account
          </a>
        </div>

        <Link className={styles.backLink} href="/">
          Back to search
        </Link>
      </main>
    </div>
  );
}
