// trunk-ignore-all(gitleaks/generic-api-key): these are public API keys

import posthog, { Properties } from "posthog-js";
import { Environment, getCurrentEnvironment } from "shared/utilities/environment";

const PRODUCTION_API_KEY = "phc_v6gcPYCrl3hssUm054zlwS1ahN4KTIwgmxie5oHCpH9";
const PRODUCTION_API_HOST = "https://ph.public.voxelplatform.com";
const DEVELOPMENT_API_KEY = "phc_56H8Eox51mjFfnGlt4fPUtUQdbJqf8HBRVvjQRyIltV";
const DEVELOPMENT_API_HOST = "https://ph-development.public.voxelplatform.com";
const EXCLUDED_EMAIL_DOMAINS = [
  "voxelai.com",
  "reviewers.voxelai.com",
  "contractors.voxelai.com",
  "uber.com",
  "ext.uber.com",
];

class AnalyticsClient {
  apiKey: string;
  apiHost: string;
  initialized: boolean = false;
  email?: string;
  organizationKey?: string;
  siteKey?: string;

  constructor() {
    const isProduction = getCurrentEnvironment() === Environment.Production;
    this.apiKey = isProduction ? PRODUCTION_API_KEY : DEVELOPMENT_API_KEY;
    this.apiHost = isProduction ? PRODUCTION_API_HOST : DEVELOPMENT_API_HOST;
  }

  private isInitialized(email: string, organizationKey: string, siteKey: string): boolean {
    // Determine if PostHog has already been initialized for this user + site combination
    return (
      this.initialized &&
      this.email === email.toLowerCase().trim() &&
      this.organizationKey === organizationKey.toLowerCase().trim() &&
      this.siteKey === siteKey.toLowerCase().trim()
    );
  }

  init(email: string, organizationKey: string, siteKey: string) {
    // Don't initialize analytics in development
    if (getCurrentEnvironment() === Environment.Development) return;

    // Already initialized
    if (this.isInitialized(email, organizationKey, siteKey)) return;

    // Excluded user
    if (excludeUser(email)) return;

    // Invalid posthog client
    if (!isPostHogClient(posthog)) return;

    // Update user properties so we know which user + site combination is being tracked
    this.email = email.toLowerCase().trim();
    this.organizationKey = organizationKey.toLowerCase().trim();
    this.siteKey = siteKey.toLowerCase().trim();
    this.initialized = true;

    // Initialize PostHog
    posthog?.init(this.apiKey, {
      api_host: this.apiHost,
      loaded: (posthog: PostHogClient) => {
        // Identify user
        posthog?.identify(email, {
          email: this.email as string,
          organizationKey: this.organizationKey as string,
          siteKey: this.siteKey as string,
          organizationSiteKey: `${this.organizationKey}:${this.siteKey}`,
        });
      },
    });
  }

  trackPageview() {
    posthog.capture("$pageview");
  }

  trackCustomEvent(eventName: string, props?: Properties) {
    posthog.capture(eventName, props);
  }

  reset() {
    try {
      posthog.reset();
    } catch (error: unknown) {
      // This has been observed to throw a TypeError in some cases where
      // the PostHog client is not properly initialized
      const ignoredError = error instanceof TypeError;
      if (!ignoredError) throw error;
    }
  }
}

export const analytics = new AnalyticsClient();

interface PostHogConfig {
  api_host: string;
  loaded: (posthog: PostHogClient) => void;
}

interface PostHogClient {
  init(apiKey: string, config: PostHogConfig): void;
  identify(email: string, properties: Record<string, string>): void;
}

function isPostHogClient(posthog: unknown): posthog is PostHogClient {
  return typeof posthog === "object" && posthog !== null && "init" in posthog && "identify" in posthog;
}

function excludeUser(email: string): boolean {
  const emailParts = email.split("@");
  const validEmail = emailParts.length === 2;
  if (!validEmail) return true;
  const domain = emailParts[1].toLowerCase().trim();
  return EXCLUDED_EMAIL_DOMAINS.includes(domain);
}
